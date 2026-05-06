import { useEffect, useRef, useState } from "react";

function convolveBernoulli(dist: number[], p: number): number[] {
  const next = Array(dist.length + 1).fill(0);
  for (let k = 0; k < dist.length; k++) {
    if (dist[k] === 0) continue;
    next[k] += dist[k] * (1 - p);
    next[k + 1] += dist[k] * p;
  }
  return next;
}

function convolve(a: number[], b: number[]): number[] {
  const res = Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) res[i + j] += a[i] * b[j];
  return res;
}

function buildPrefixSuffix(p: number[]) {
  const n = p.length;
  const prefix: number[][] = Array(n + 1);
  const suffix: number[][] = Array(n + 1);
  prefix[0] = [1];
  for (let i = 0; i < n; i++) prefix[i + 1] = convolveBernoulli(prefix[i], p[i]);
  suffix[n] = [1];
  for (let i = n - 1; i >= 0; i--) suffix[i] = convolveBernoulli(suffix[i + 1], p[i]);
  return { prefix, suffix };
}

export function computePrices(p: number[]) {
  const n = p.length;
  let priceY = 1;
  for (const pi of p) priceY *= 1 - pi;
  const { prefix, suffix } = buildPrefixSuffix(p);
  const prices = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const dist = convolve(prefix[i], suffix[i + 1]);
    let expectation = 0;
    for (let k = 0; k < dist.length; k++) expectation += dist[k] * (1 / (1 + k));
    prices[i] = p[i] * expectation;
  }
  return { priceY, prices };
}

// ─── Worker blob ──────────────────────────────────────────────────────────────
// The solver runs entirely off the main thread so the UI never freezes.

const WORKER_SRC = `
function convolveBernoulli(dist, p) {
  const next = Array(dist.length + 1).fill(0);

  for (let k = 0; k < dist.length; k++) {
    if (dist[k] === 0) continue;

    next[k] += dist[k] * (1 - p);
    next[k + 1] += dist[k] * p;
  }

  return next;
}

function convolve(a, b) {
  const res = Array(a.length + b.length - 1).fill(0);

  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      res[i + j] += a[i] * b[j];
    }
  }

  return res;
}

function buildPrefixSuffix(p) {
  const n = p.length;

  const prefix = Array(n + 1);
  const suffix = Array(n + 1);

  prefix[0] = [1];

  for (let i = 0; i < n; i++) {
    prefix[i + 1] = convolveBernoulli(prefix[i], p[i]);
  }

  suffix[n] = [1];

  for (let i = n - 1; i >= 0; i--) {
    suffix[i] = convolveBernoulli(suffix[i + 1], p[i]);
  }

  return { prefix, suffix };
}

function computePrices(p) {
  const n = p.length;

  let priceY = 1;

  for (const pi of p) {
    priceY *= 1 - pi;
  }

  const { prefix, suffix } = buildPrefixSuffix(p);

  const prices = Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    const dist = convolve(prefix[i], suffix[i + 1]);

    let expectation = 0;

    for (let k = 0; k < dist.length; k++) {
      expectation += dist[k] * (1 / (1 + k));
    }

    prices[i] = p[i] * expectation;
  }

  return { priceY, prices };
}

function solveProbs(targetY, targetPrices, lr, iters) {
  const n = targetPrices.length;

  // Scale eps to the magnitude of the smallest target
  const minTarget = Math.min(targetY, ...targetPrices);
  const eps = Math.max(1e-8, minTarget * 1e-3);

  // Better init: prices underestimate probs, so inflate by ~2x capped
  let p = targetPrices.map(v =>
    Math.max(1e-6, Math.min(1 - 1e-6, v * 2))
  );

  function loss(p) {
    const { priceY, prices } = computePrices(p);
    let err = (priceY - targetY) ** 2;
    for (let i = 0; i < n; i++) err += (prices[i] - targetPrices[i]) ** 2;
    return err;
  }

  const m = Array(n).fill(0);
  const v = Array(n).fill(0);
  const beta1 = 0.9, beta2 = 0.999, adamEps = 1e-8;

  let prevLoss = Infinity;

  for (let iter = 1; iter <= iters; iter++) {
    const base = loss(p);

    // Exit threshold scales with the loss magnitude, not absolute
    if (iter > 1 && Math.abs(prevLoss - base) < base * 1e-9) break;
    prevLoss = base;

    const grad = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const up = [...p], down = [...p];
      up[i]   = Math.min(1 - 1e-9, p[i] + eps);
      down[i] = Math.max(1e-9,     p[i] - eps);
      grad[i] = (loss(up) - loss(down)) / (up[i] - down[i]); // use actual step
    }

    let gradNorm = 0;
    for (let i = 0; i < n; i++) gradNorm += grad[i] * grad[i];
    gradNorm = Math.sqrt(gradNorm);

    // Remove the sqrt(base) amplification — it destabilizes small-loss regime
    const stepScale = gradNorm > 1e-15 ? lr / gradNorm : lr;

    const candidate = [...p];
    for (let i = 0; i < n; i++) {
      m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
      v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
      const mHat = m[i] / (1 - beta1 ** iter);
      const vHat = v[i] / (1 - beta2 ** iter);
      candidate[i] -= stepScale * mHat / (Math.sqrt(vHat) + adamEps);
      candidate[i] = Math.max(1e-9, Math.min(1 - 1e-9, candidate[i]));
    }

    const candidateLoss = loss(candidate);
    if (candidateLoss < base) {
      p = candidate;
    } else {
      let improved = false;
      for (let shrink = 0.5; shrink > 1e-6; shrink *= 0.5) {
        const trial = p.map((pi, i) => {
          const mHat = m[i] / (1 - beta1 ** iter);
          const vHat = v[i] / (1 - beta2 ** iter);
          const stepped = pi - shrink * stepScale * mHat / (Math.sqrt(vHat) + adamEps);
          return Math.max(1e-9, Math.min(1 - 1e-9, stepped));
        });
        if (loss(trial) < base) { p = trial; improved = true; break; }
      }
      if (!improved) {
        // Noise proportional to current scale, not fixed 1e-4
        for (let i = 0; i < n; i++) {
          p[i] += (Math.random() - 0.5) * Math.max(1e-7, p[i] * 1e-3);
          p[i] = Math.max(1e-9, Math.min(1 - 1e-9, p[i]));
        }
      }
    }

    if (iter % 20 === 0) self.postMessage({ type: 'progress', iter, iters, loss: base });
  }

  return p;
}

self.onmessage = function(e) {
  const {
    targetY,
    targetPrices,
    lr,
    iters,
  } = e.data;

  const probs = solveProbs(
    targetY,
    targetPrices,
    lr,
    iters
  );

  const {
    priceY,
    prices,
  } = computePrices(probs);

  self.postMessage({
    type: 'done',
    probs,
    priceY,
    prices,
  });
};
`;

function makeWorker() {
  const blob = new Blob([WORKER_SRC], { type: "application/javascript" });

  return new Worker(URL.createObjectURL(blob));
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

type SolverState =
  | { status: "idle" }
  | { status: "running"; progress: number; loss: number }
  | { status: "done"; probs: number[]; priceY: number; prices: number[] }
  | { status: "error"; message: string };

export function useImpliedProbsAsync(
  targetY: number,
  targetPrices: number[],
  enabled: boolean,
  opts?: { lr?: number; iters?: number },
): SolverState {
  const [state, setState] = useState<SolverState>({ status: "idle" });
  const workerRef = useRef<Worker | null>(null);
  const key = `${targetY}|${targetPrices.join(",")}|${opts?.lr}|${opts?.iters}`;

  useEffect(() => {
    if (!enabled) {
      setState({ status: "idle" });
      return;
    }
    if (workerRef.current) workerRef.current.terminate();

    const worker = makeWorker();
    workerRef.current = worker;
    setState({ status: "running", progress: 0, loss: Infinity });

    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "progress") {
        setState({
          status: "running",
          progress: msg.iter / msg.iters,
          loss: msg.loss,
        });
      } else if (msg.type === "done") {
        setState({
          status: "done",
          probs: msg.probs,
          priceY: msg.priceY,
          prices: msg.prices,
        });
        worker.terminate();
      }
    };
    worker.onerror = (e) => {
      setState({ status: "error", message: e.message });
      worker.terminate();
    };

    worker.postMessage({
      targetY,
      targetPrices,
      lr: opts?.lr ?? 0.01,
      iters: opts?.iters ?? 3000,
    });

    return () => {
      worker.terminate();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);

  return state;
}

export function solveProbsAsync(
  targetY: number,
  targetPrices: number[],
  opts?: {
    lr?: number;
    iters?: number;
    onProgress?: (progress: {
      iter: number;
      iters: number;
      progress: number;
      loss: number;
    }) => void;
  },
): Promise<{
  probs: number[];
  priceY: number;
  prices: number[];
}> {
  return new Promise((resolve, reject) => {
    const worker = makeWorker();

    worker.onmessage = (e) => {
      const msg = e.data;

      if (msg.type === "progress") {
        opts?.onProgress?.({
          iter: msg.iter,
          iters: msg.iters,
          progress: msg.iter / msg.iters,
          loss: msg.loss,
        });
      } else if (msg.type === "done") {
        worker.terminate();

        resolve({
          probs: msg.probs,
          priceY: msg.priceY,
          prices: msg.prices,
        });
      }
    };

    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message));
    };

    worker.postMessage({
      targetY,
      targetPrices,
      lr: opts?.lr ?? 0.01,
      iters: opts?.iters ?? 100,
    });
  });
}
