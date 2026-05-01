import { useState, useMemo, useCallback } from "react";

// ─── Math core ────────────────────────────────────────────────────────────────

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

function computePrices(p: number[]) {
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

function solveProbs(
  targetY: number,
  targetPrices: number[],
  opts?: { lr?: number; iters?: number },
): number[] {
  const n = targetPrices.length;
  let p = Array(n).fill(0.5);
  const lr = opts?.lr ?? 0.05;
  const iters = opts?.iters ?? 4000;
  const eps = 1e-6;

  function loss(p: number[]) {
    const { priceY, prices } = computePrices(p);
    let err = (priceY - targetY) ** 2;
    for (let i = 0; i < n; i++) err += (prices[i] - targetPrices[i]) ** 2;
    return err;
  }

  let prevLoss = Infinity;
  for (let iter = 0; iter < iters; iter++) {
    const base = loss(p);
    if (Math.abs(prevLoss - base) < 1e-12) break;
    prevLoss = base;
    const grad = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const p2 = [...p];
      p2[i] += eps;
      grad[i] = (loss(p2) - base) / eps;
    }
    for (let i = 0; i < n; i++) {
      p[i] -= lr * grad[i];
      p[i] = Math.max(1e-6, Math.min(1 - 1e-6, p[i]));
    }
  }
  return p;
}

function useImpliedProbs(
  targetY: number,
  targetPrices: number[],
  opts?: { lr?: number; iters?: number },
) {
  const probs = useMemo(
    () => solveProbs(targetY, targetPrices, opts),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [targetY, targetPrices.join(","), opts?.lr, opts?.iters],
  );
  const { priceY, prices } = useMemo(() => computePrices(probs), [probs]);
  return { probs, priceY, prices };
}

// ─── Types ────────────────────────────────────────────────────────────────────

type Mode = "pricesToProbs" | "probsToPrices";

const LABELS = ["A", "B", "C", "D", "E", "F"];

// ─── Components ───────────────────────────────────────────────────────────────

function NumInput({
  value,
  onChange,
  label,
  disabled = false,
}: {
  value: string;
  onChange: (v: string) => void;
  label: string;
  disabled?: boolean;
}) {
  const num = parseFloat(value);
  const valid = !isNaN(num) && num >= 0 && num <= 1;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <label
        style={{
          fontSize: 11,
          fontWeight: 600,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "var(--muted)",
          fontFamily: "var(--mono)",
        }}
      >
        {label}
      </label>
      <div style={{ position: "relative" }}>
        <input
          type="number"
          min={0}
          max={1}
          step={0.01}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          style={{
            width: "100%",
            padding: "10px 14px",
            background: disabled ? "var(--input-disabled)" : "var(--input-bg)",
            border: `1.5px solid ${!valid && value !== "" ? "var(--error)" : "var(--border)"}`,
            borderRadius: 8,
            color: disabled ? "var(--muted)" : "var(--fg)",
            fontSize: 15,
            fontFamily: "var(--mono)",
            fontWeight: 500,
            outline: "none",
            transition: "border-color 0.15s, background 0.15s",
            boxSizing: "border-box",
            cursor: disabled ? "not-allowed" : "text",
          }}
          onFocus={(e) => {
            if (!disabled) (e.target as HTMLInputElement).style.borderColor = "var(--accent)";
          }}
          onBlur={(e) => {
            (e.target as HTMLInputElement).style.borderColor =
              !valid && value !== "" ? "var(--error)" : "var(--border)";
          }}
        />
        {disabled && (
          <div
            style={{
              position: "absolute",
              right: 12,
              top: "50%",
              transform: "translateY(-50%)",
              fontSize: 10,
              fontFamily: "var(--mono)",
              color: "var(--muted)",
              letterSpacing: "0.05em",
            }}
          >
            computed
          </div>
        )}
      </div>
    </div>
  );
}

function ResultBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <span
        style={{
          width: 28,
          fontSize: 12,
          fontFamily: "var(--mono)",
          fontWeight: 700,
          color: "var(--muted)",
          letterSpacing: "0.08em",
          flexShrink: 0,
        }}
      >
        {label}
      </span>
      <div
        style={{
          flex: 1,
          height: 8,
          background: "var(--track)",
          borderRadius: 99,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${value * 100}%`,
            background: color,
            borderRadius: 99,
            transition: "width 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)",
          }}
        />
      </div>
      <span
        style={{
          width: 52,
          textAlign: "right",
          fontSize: 14,
          fontFamily: "var(--mono)",
          fontWeight: 600,
          color: "var(--fg)",
          flexShrink: 0,
        }}
      >
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

// ─── Prices → Probs mode ──────────────────────────────────────────────────────

function PricesToProbs({ n }: { n: number }) {
  const [priceY, setPriceY] = useState("0.10");
  const [eventPrices, setEventPrices] = useState<string[]>(
    Array(n)
      .fill("")
      .map((_, i) => ["0.20", "0.40", "0.30", "0.15", "0.10", "0.08"][i] ?? "0.10"),
  );

  const setPrice = useCallback((i: number, v: string) => {
    setEventPrices((prev) => {
      const next = [...prev];
      next[i] = v;
      return next;
    });
  }, []);

  const targetY = parseFloat(priceY);
  const targetPrices = eventPrices.map(parseFloat);
  const allValid =
    !isNaN(targetY) &&
    targetY > 0 &&
    targetY <= 1 &&
    targetPrices.every((p) => !isNaN(p) && p > 0 && p <= 1);

  const {
    probs,
    priceY: computedPriceY,
    prices: computedPrices,
  } = useImpliedProbs(
    allValid ? targetY : 0.1,
    allValid ? targetPrices.slice(0, n) : Array(n).fill(0.1),
  );

  const COLORS = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

  const sumCheck = allValid ? targetY + targetPrices.slice(0, n).reduce((a, b) => a + b, 0) : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      {/* Input */}
      <div>
        <p
          style={{
            fontSize: 12,
            color: "var(--muted)",
            marginBottom: 16,
            fontFamily: "var(--mono)",
            letterSpacing: "0.05em",
          }}
        >
          ENTER MARKET PRICES (0–1)
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div style={{ gridColumn: "1 / -1" }}>
            <NumInput label='Y — "none happen"' value={priceY} onChange={setPriceY} />
          </div>
          {Array.from({ length: n }, (_, i) => (
            <NumInput
              key={i}
              label={`Event ${LABELS[i]}`}
              value={eventPrices[i] ?? ""}
              onChange={(v) => setPrice(i, v)}
            />
          ))}
        </div>

        {allValid && (
          <div
            style={{
              marginTop: 12,
              padding: "8px 12px",
              background: Math.abs(sumCheck - 1) < 0.02 ? "var(--success-bg)" : "var(--warn-bg)",
              border: `1px solid ${Math.abs(sumCheck - 1) < 0.02 ? "var(--success)" : "var(--warn)"}`,
              borderRadius: 8,
              fontSize: 12,
              fontFamily: "var(--mono)",
              color: Math.abs(sumCheck - 1) < 0.02 ? "var(--success)" : "var(--warn)",
            }}
          >
            Σ prices = {sumCheck.toFixed(4)}{" "}
            {Math.abs(sumCheck - 1) < 0.02 ? "✓ sum ≈ 1" : "— sum ≠ 1, may not converge well"}
          </div>
        )}
      </div>

      {/* Output */}
      {allValid && (
        <div>
          <p
            style={{
              fontSize: 12,
              color: "var(--muted)",
              marginBottom: 16,
              fontFamily: "var(--mono)",
              letterSpacing: "0.05em",
            }}
          >
            IMPLIED PROBABILITIES
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {probs.map((prob, i) => (
              <ResultBar key={i} label={LABELS[i]} value={prob} color={COLORS[i % COLORS.length]} />
            ))}
          </div>

          <div style={{ marginTop: 20, borderTop: "1px solid var(--border)", paddingTop: 16 }}>
            <p
              style={{
                fontSize: 11,
                color: "var(--muted)",
                marginBottom: 12,
                fontFamily: "var(--mono)",
                letterSpacing: "0.08em",
              }}
            >
              VERIFICATION — PRICES BACK FROM PROBS
            </p>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(110px, 1fr))",
                gap: 8,
              }}
            >
              {[
                ["Y", computedPriceY],
                ...computedPrices.map((p, i) => [LABELS[i], p] as [string, number]),
              ].map(([label, val]) => (
                <div
                  key={label as string}
                  style={{
                    background: "var(--card2)",
                    borderRadius: 8,
                    padding: "8px 12px",
                    display: "flex",
                    flexDirection: "column",
                    gap: 3,
                  }}
                >
                  <span
                    style={{
                      fontSize: 10,
                      fontFamily: "var(--mono)",
                      color: "var(--muted)",
                      letterSpacing: "0.1em",
                    }}
                  >
                    {label as string}
                  </span>
                  <span
                    style={{
                      fontSize: 14,
                      fontFamily: "var(--mono)",
                      fontWeight: 600,
                      color: "var(--fg)",
                    }}
                  >
                    {(val as number).toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Probs → Prices mode ──────────────────────────────────────────────────────

function ProbsToPrices({ n }: { n: number }) {
  const [probs, setProbs] = useState<string[]>(
    Array(n)
      .fill("")
      .map((_, i) => ["0.382", "0.655", "0.531", "0.40", "0.25", "0.15"][i] ?? "0.30"),
  );

  const setProb = useCallback((i: number, v: string) => {
    setProbs((prev) => {
      const next = [...prev];
      next[i] = v;
      return next;
    });
  }, []);

  const numProbs = probs.map(parseFloat);
  const allValid = numProbs.every((p) => !isNaN(p) && p > 0 && p < 1);

  const { priceY, prices } = useMemo(
    () => (allValid ? computePrices(numProbs.slice(0, n)) : { priceY: 0, prices: [] }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [allValid, numProbs.join(","), n],
  );

  const COLORS = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <div>
        <p
          style={{
            fontSize: 12,
            color: "var(--muted)",
            marginBottom: 16,
            fontFamily: "var(--mono)",
            letterSpacing: "0.05em",
          }}
        >
          ENTER PROBABILITIES (0–1)
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          {Array.from({ length: n }, (_, i) => (
            <NumInput
              key={i}
              label={`P(${LABELS[i]})`}
              value={probs[i] ?? ""}
              onChange={(v) => setProb(i, v)}
            />
          ))}
        </div>
      </div>

      {allValid && (
        <div>
          <p
            style={{
              fontSize: 12,
              color: "var(--muted)",
              marginBottom: 16,
              fontFamily: "var(--mono)",
              letterSpacing: "0.05em",
            }}
          >
            COMPUTED MARKET PRICES
          </p>

          <div
            style={{
              background: "var(--card2)",
              borderRadius: 10,
              padding: "12px 16px",
              marginBottom: 12,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span
              style={{
                fontSize: 12,
                fontFamily: "var(--mono)",
                color: "var(--muted)",
                letterSpacing: "0.08em",
              }}
            >
              Y — NONE HAPPEN
            </span>
            <span
              style={{
                fontSize: 20,
                fontFamily: "var(--mono)",
                fontWeight: 700,
                color: "var(--fg)",
              }}
            >
              {priceY.toFixed(4)}
            </span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {prices.map((price, i) => (
              <ResultBar
                key={i}
                label={LABELS[i]}
                value={price}
                color={COLORS[i % COLORS.length]}
              />
            ))}
          </div>

          <div
            style={{
              marginTop: 16,
              padding: "10px 14px",
              background: "var(--card2)",
              borderRadius: 8,
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <span style={{ fontSize: 12, fontFamily: "var(--mono)", color: "var(--muted)" }}>
              Σ all prices
            </span>
            <span
              style={{
                fontSize: 13,
                fontFamily: "var(--mono)",
                fontWeight: 600,
                color: "var(--fg)",
              }}
            >
              {(priceY + prices.reduce((a, b) => a + b, 0)).toFixed(4)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [mode, setMode] = useState<Mode>("pricesToProbs");
  const [n, setN] = useState(3);

  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #0d0d0f;
          --card: #16161a;
          --card2: #1e1e24;
          --border: #2a2a32;
          --fg: #e8e8f0;
          --muted: #5a5a72;
          --accent: #6366f1;
          --accent2: #06b6d4;
          --input-bg: #121216;
          --input-disabled: #0f0f12;
          --track: #1e1e26;
          --error: #ef4444;
          --warn: #f59e0b;
          --warn-bg: rgba(245,158,11,0.08);
          --success: #10b981;
          --success-bg: rgba(16,185,129,0.08);
          --mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
          --display: 'Syne', sans-serif;
        }

        html, body, #root {
          min-height: 100vh;
          background: var(--bg);
          color: var(--fg);
        }

        body {
          font-family: var(--display);
          -webkit-font-smoothing: antialiased;
        }

        input[type=number] {
          -moz-appearance: textfield;
          appearance: textfield;
        }
        input[type=number]::-webkit-outer-spin-button,
        input[type=number]::-webkit-inner-spin-button {
          -webkit-appearance: none;
        }

        .tab-btn {
          flex: 1;
          padding: 10px 0;
          background: transparent;
          border: none;
          border-radius: 8px;
          font-family: var(--mono);
          font-size: 12px;
          font-weight: 600;
          letter-spacing: 0.08em;
          cursor: pointer;
          transition: background 0.15s, color 0.15s;
          color: var(--muted);
        }
        .tab-btn.active {
          background: var(--accent);
          color: #fff;
        }
        .tab-btn:hover:not(.active) {
          background: var(--card2);
          color: var(--fg);
        }

        .n-btn {
          width: 32px;
          height: 32px;
          border-radius: 6px;
          border: 1.5px solid var(--border);
          background: transparent;
          color: var(--fg);
          font-family: var(--mono);
          font-size: 14px;
          font-weight: 700;
          cursor: pointer;
          transition: background 0.12s, border-color 0.12s;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .n-btn.selected {
          background: var(--card2);
          border-color: var(--accent);
          color: var(--accent);
        }
        .n-btn:hover:not(.selected) {
          background: var(--card2);
        }

        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
      `}</style>

      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "center",
          padding: "48px 16px 80px",
        }}
      >
        <div style={{ width: "100%", maxWidth: 480 }}>
          {/* Header */}
          <div style={{ marginBottom: 36 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: "var(--accent)",
                  boxShadow: "0 0 12px var(--accent)",
                }}
              />
              <span
                style={{
                  fontSize: 11,
                  fontFamily: "var(--mono)",
                  color: "var(--muted)",
                  letterSpacing: "0.15em",
                }}
              >
                PREDICTION MARKET
              </span>
            </div>
            <h1
              style={{
                fontSize: 32,
                fontWeight: 800,
                letterSpacing: "-0.02em",
                lineHeight: 1.1,
                background: "linear-gradient(135deg, #e8e8f0 0%, #6366f1 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              Implied Probability
              <br />
              Solver
            </h1>
            <p style={{ marginTop: 10, fontSize: 14, color: "var(--muted)", lineHeight: 1.6 }}>
              Convert between market prices and implied event probabilities.
            </p>
          </div>

          {/* Card */}
          <div
            style={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 16,
              padding: 24,
              display: "flex",
              flexDirection: "column",
              gap: 24,
            }}
          >
            {/* Mode tabs */}
            <div
              style={{
                display: "flex",
                gap: 4,
                background: "var(--input-bg)",
                padding: 4,
                borderRadius: 10,
              }}
            >
              <button
                className={`tab-btn ${mode === "pricesToProbs" ? "active" : ""}`}
                onClick={() => setMode("pricesToProbs")}
              >
                PRICES → PROBS
              </button>
              <button
                className={`tab-btn ${mode === "probsToPrices" ? "active" : ""}`}
                onClick={() => setMode("probsToPrices")}
              >
                PROBS → PRICES
              </button>
            </div>

            {/* Event count */}
            <div>
              <p
                style={{
                  fontSize: 11,
                  fontFamily: "var(--mono)",
                  color: "var(--muted)",
                  letterSpacing: "0.1em",
                  marginBottom: 10,
                }}
              >
                NUMBER OF EVENTS
              </p>
              <div style={{ display: "flex", gap: 6 }}>
                {[2, 3, 4, 5, 6].map((v) => (
                  <button
                    key={v}
                    className={`n-btn ${n === v ? "selected" : ""}`}
                    onClick={() => setN(v)}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ height: 1, background: "var(--border)" }} />

            {/* Panel */}
            {mode === "pricesToProbs" ? (
              <PricesToProbs key={`p2p-${n}`} n={n} />
            ) : (
              <ProbsToPrices key={`pr2p-${n}`} n={n} />
            )}
          </div>

          {/* Footer */}
          <p
            style={{
              marginTop: 20,
              textAlign: "center",
              fontSize: 11,
              fontFamily: "var(--mono)",
              color: "var(--muted)",
              letterSpacing: "0.05em",
            }}
          >
            gradient descent · {4000} iters · lr 0.05
          </p>
        </div>
      </div>
    </>
  );
}
