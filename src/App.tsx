import { useCallback, useMemo, useState } from "react";
import { computePrices, useImpliedProbsAsync } from "./useImpliedProbs";

// ─── UI ───────────────────────────────────────────────────────────────────────

const COLORS = [
  "#6366f1",
  "#06b6d4",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#14b8a6",
  "#f97316",
  "#84cc16",
  "#3b82f6",
  "#a78bfa",
];

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
  const valid = !isNaN(parseFloat(value)) && parseFloat(value) > 0 && parseFloat(value) <= 1;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <label
        style={{
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "var(--muted)",
          fontFamily: "var(--mono)",
        }}
      >
        {label}
      </label>
      <input
        type="number"
        min={0}
        max={1}
        step={0.001}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%",
          padding: "8px 10px",
          background: disabled ? "var(--input-disabled)" : "var(--input-bg)",
          border: `1.5px solid ${!valid && value !== "" ? "var(--error)" : "var(--border)"}`,
          borderRadius: 7,
          color: disabled ? "var(--muted)" : "var(--fg)",
          fontSize: 13,
          fontFamily: "var(--mono)",
          fontWeight: 500,
          outline: "none",
          boxSizing: "border-box",
        }}
      />
    </div>
  );
}

function ResultBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span
        style={{
          width: 40,
          fontSize: 10,
          fontFamily: "var(--mono)",
          color: "var(--muted)",
          flexShrink: 0,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </span>
      <div
        style={{
          flex: 1,
          height: 5,
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
            transition: "width 0.3s ease",
          }}
        />
      </div>
      <span
        style={{
          width: 48,
          textAlign: "right",
          fontSize: 12,
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

function ProgressBar({ progress, loss }: { progress: number; loss: number }) {
  const lossStr = loss === Infinity ? "…" : loss < 1e-10 ? loss.toExponential(2) : loss.toFixed(10);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: "14px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span
          style={{
            fontSize: 11,
            fontFamily: "var(--mono)",
            color: "var(--muted)",
            letterSpacing: "0.08em",
          }}
        >
          SOLVING…
        </span>
        <span style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--accent)" }}>
          {Math.round(progress * 100)}%
        </span>
      </div>
      <div style={{ height: 4, background: "var(--track)", borderRadius: 99, overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: `${progress * 100}%`,
            background: "linear-gradient(90deg, var(--accent), var(--accent2))",
            borderRadius: 99,
            transition: "width 0.2s ease",
          }}
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--muted)" }}>
          loss: {lossStr}
        </span>
        <span style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--muted)" }}>
          analytical gradient · adam
        </span>
      </div>
    </div>
  );
}

function BulkInput({
  onParsed,
  n,
}: {
  onParsed: (y: string, prices: string[]) => void;
  n: number;
}) {
  const [raw, setRaw] = useState("");
  const [error, setError] = useState("");

  function parse() {
    const nums = raw
      .trim()
      .split(/[\s,;\n]+/)
      .map(Number)
      .filter((x) => !isNaN(x));
    if (nums.length < n + 1) {
      setError(`Need ${n + 1} numbers (Y + ${n} events), got ${nums.length}`);
      return;
    }
    setError("");
    onParsed(String(nums[0]), nums.slice(1, n + 1).map(String));
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <label
        style={{
          fontSize: 11,
          fontWeight: 600,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--muted)",
          fontFamily: "var(--mono)",
        }}
      >
        PASTE PRICES{" "}
        <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>
          (Y first, then events)
        </span>
      </label>
      <textarea
        rows={3}
        placeholder={`0.000315, 0.0167, 0.0176, ...`}
        value={raw}
        onChange={(e) => setRaw(e.target.value)}
        style={{
          background: "var(--input-bg)",
          border: "1.5px solid var(--border)",
          borderRadius: 8,
          color: "var(--fg)",
          fontSize: 12,
          fontFamily: "var(--mono)",
          padding: "10px 12px",
          resize: "vertical",
          lineHeight: 1.6,
        }}
      />
      {error && (
        <span style={{ fontSize: 11, color: "var(--error)", fontFamily: "var(--mono)" }}>
          {error}
        </span>
      )}
      <button
        onClick={parse}
        style={{
          padding: "7px 14px",
          background: "var(--card2)",
          border: "1px solid var(--border)",
          borderRadius: 7,
          color: "var(--fg)",
          fontSize: 11,
          fontFamily: "var(--mono)",
          fontWeight: 700,
          letterSpacing: "0.08em",
          cursor: "pointer",
          alignSelf: "flex-start",
        }}
      >
        LOAD →
      </button>
    </div>
  );
}

// ─── Prices → Probs ───────────────────────────────────────────────────────────

function PricesToProbs({ n, lr, iters }: { n: number; lr: number; iters: number }) {
  const defaultPrices = ["0.20", "0.40", "0.30", "0.15", "0.10", "0.08"];
  const [priceY, setPriceY] = useState("0.10");
  const [eventPrices, setEventPrices] = useState<string[]>(
    Array(n)
      .fill("")
      .map((_, i) => defaultPrices[i] ?? "0.10"),
  );
  const [submitted, setSubmitted] = useState(false);

  const setPrice = useCallback((i: number, v: string) => {
    setEventPrices((prev) => {
      const next = [...prev];
      next[i] = v;
      return next;
    });
    setSubmitted(false);
  }, []);

  const targetY = parseFloat(priceY);
  const targetPrices = eventPrices.slice(0, n).map(parseFloat);
  const allValid =
    !isNaN(targetY) &&
    targetY > 0 &&
    targetY <= 1 &&
    targetPrices.length === n &&
    targetPrices.every((p) => !isNaN(p) && p > 0 && p <= 1);

  const state = useImpliedProbsAsync(targetY, targetPrices, allValid && submitted, { lr, iters });
  const sumCheck = allValid ? targetY + targetPrices.reduce((a, b) => a + b, 0) : 0;
  const sumOk = Math.abs(sumCheck - 1) < 0.02;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <BulkInput
        n={n}
        onParsed={(y, prices) => {
          setPriceY(y);
          setEventPrices(prices);
          setSubmitted(false);
        }}
      />

      <div style={{ height: 1, background: "var(--border)" }} />

      {n <= 20 && (
        <div>
          <p
            style={{
              fontSize: 11,
              color: "var(--muted)",
              marginBottom: 12,
              fontFamily: "var(--mono)",
              letterSpacing: "0.05em",
            }}
          >
            OR ENTER MANUALLY
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: n <= 6 ? "1fr 1fr" : "1fr 1fr 1fr",
              gap: 8,
            }}
          >
            <div style={{ gridColumn: "1 / -1" }}>
              <NumInput
                label='Y — "none happen"'
                value={priceY}
                onChange={(v) => {
                  setPriceY(v);
                  setSubmitted(false);
                }}
              />
            </div>
            {Array.from({ length: n }, (_, i) => (
              <NumInput
                key={i}
                label={`Event ${i + 1}`}
                value={eventPrices[i] ?? ""}
                onChange={(v) => setPrice(i, v)}
              />
            ))}
          </div>
        </div>
      )}

      {allValid && (
        <div
          style={{
            padding: "7px 12px",
            borderRadius: 7,
            fontSize: 11,
            fontFamily: "var(--mono)",
            background: sumOk ? "var(--success-bg)" : "var(--warn-bg)",
            border: `1px solid ${sumOk ? "var(--success)" : "var(--warn)"}`,
            color: sumOk ? "var(--success)" : "var(--warn)",
          }}
        >
          Σ prices = {sumCheck.toFixed(5)}{" "}
          {sumOk ? "✓ looks good" : "— sum ≠ 1, convergence may suffer"}
        </div>
      )}

      {allValid && !submitted && (
        <button
          onClick={() => setSubmitted(true)}
          style={{
            padding: "12px",
            background: "var(--accent)",
            border: "none",
            borderRadius: 10,
            color: "#fff",
            fontSize: 13,
            fontFamily: "var(--mono)",
            fontWeight: 700,
            letterSpacing: "0.08em",
            cursor: "pointer",
            boxShadow: "0 0 20px rgba(99,102,241,0.25)",
          }}
        >
          SOLVE →
        </button>
      )}

      {submitted && state.status === "running" && (
        <ProgressBar progress={state.progress} loss={state.loss} />
      )}

      {submitted &&
        state.status === "done" &&
        (() => {
          const maxErr = Math.max(
            ...state.prices.map((p, i) => Math.abs(p - targetPrices[i])),
            Math.abs(state.priceY - targetY),
          );
          const converged = maxErr < 1e-4;
          return (
            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 12,
                }}
              >
                <p
                  style={{
                    fontSize: 11,
                    color: "var(--muted)",
                    fontFamily: "var(--mono)",
                    letterSpacing: "0.05em",
                  }}
                >
                  IMPLIED PROBABILITIES
                </p>
                <span
                  style={{
                    fontSize: 10,
                    fontFamily: "var(--mono)",
                    color: converged ? "var(--success)" : "var(--warn)",
                    padding: "3px 8px",
                    borderRadius: 99,
                    border: `1px solid ${converged ? "var(--success)" : "var(--warn)"}`,
                  }}
                >
                  {converged
                    ? `✓ max err ${maxErr.toExponential(1)}`
                    : `⚠ max err ${maxErr.toExponential(1)} — try more iters`}
                </span>
              </div>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                  maxHeight: 420,
                  overflowY: "auto",
                  paddingRight: 4,
                }}
              >
                {state.probs.map((prob, i) => (
                  <ResultBar
                    key={i}
                    label={`E${i + 1}`}
                    value={prob}
                    color={COLORS[i % COLORS.length]}
                  />
                ))}
              </div>

              <div style={{ marginTop: 16, borderTop: "1px solid var(--border)", paddingTop: 14 }}>
                <p
                  style={{
                    fontSize: 10,
                    color: "var(--muted)",
                    marginBottom: 10,
                    fontFamily: "var(--mono)",
                    letterSpacing: "0.08em",
                  }}
                >
                  VERIFICATION — PRICES RECONSTRUCTED FROM SOLVED PROBS
                </p>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(84px, 1fr))",
                    gap: 6,
                    maxHeight: 200,
                    overflowY: "auto",
                  }}
                >
                  {[
                    ["Y", state.priceY, targetY],
                    ...state.prices.map(
                      (p, i) => [`E${i + 1}`, p, targetPrices[i]] as [string, number, number],
                    ),
                  ].map(([label, val, target]) => {
                    const err = Math.abs((val as number) - (target as number));
                    return (
                      <div
                        key={label as string}
                        style={{
                          background: "var(--card2)",
                          borderRadius: 7,
                          padding: "7px 10px",
                          border: `1px solid ${err < 1e-4 ? "transparent" : "var(--warn)"}`,
                        }}
                      >
                        <div
                          style={{
                            fontSize: 9,
                            fontFamily: "var(--mono)",
                            color: "var(--muted)",
                            marginBottom: 2,
                          }}
                        >
                          {label as string}
                        </div>
                        <div
                          style={{
                            fontSize: 12,
                            fontFamily: "var(--mono)",
                            fontWeight: 600,
                            color: "var(--fg)",
                          }}
                        >
                          {(val as number).toFixed(5)}
                        </div>
                        <div
                          style={{
                            fontSize: 9,
                            fontFamily: "var(--mono)",
                            color: err < 1e-4 ? "var(--success)" : "var(--warn)",
                          }}
                        >
                          Δ {err.toExponential(1)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <button
                onClick={() => setSubmitted(false)}
                style={{
                  marginTop: 14,
                  padding: "7px 14px",
                  background: "transparent",
                  border: "1px solid var(--border)",
                  borderRadius: 7,
                  color: "var(--muted)",
                  fontSize: 11,
                  fontFamily: "var(--mono)",
                  cursor: "pointer",
                }}
              >
                ← EDIT PRICES
              </button>
            </div>
          );
        })()}

      {state.status === "error" && (
        <div
          style={{
            padding: "10px 14px",
            background: "rgba(239,68,68,0.1)",
            border: "1px solid var(--error)",
            borderRadius: 8,
            fontSize: 12,
            fontFamily: "var(--mono)",
            color: "var(--error)",
          }}
        >
          Worker error: {(state as any).message}
        </div>
      )}
    </div>
  );
}

// ─── Probs → Prices ───────────────────────────────────────────────────────────

function ProbsToPrices({ n }: { n: number }) {
  const defaults = ["0.382", "0.655", "0.531", "0.40", "0.25", "0.15"];
  const [probs, setProbs] = useState<string[]>(
    Array(n)
      .fill("")
      .map((_, i) => defaults[i] ?? "0.30"),
  );
  const setProb = useCallback((i: number, v: string) => {
    setProbs((prev) => {
      const next = [...prev];
      next[i] = v;
      return next;
    });
  }, []);

  const numProbs = probs.slice(0, n).map(parseFloat);
  const allValid = numProbs.length === n && numProbs.every((p) => !isNaN(p) && p > 0 && p < 1);
  const { priceY, prices } = useMemo(
    () => (allValid ? computePrices(numProbs) : { priceY: 0, prices: [] }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [allValid, numProbs.join(",")],
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {n <= 20 ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: n <= 6 ? "1fr 1fr" : "1fr 1fr 1fr",
            gap: 8,
          }}
        >
          {Array.from({ length: n }, (_, i) => (
            <NumInput
              key={i}
              label={`P(event ${i + 1})`}
              value={probs[i] ?? ""}
              onChange={(v) => setProb(i, v)}
            />
          ))}
        </div>
      ) : (
        <div
          style={{
            padding: "12px",
            background: "var(--card2)",
            borderRadius: 8,
            fontSize: 12,
            fontFamily: "var(--mono)",
            color: "var(--muted)",
          }}
        >
          Manual input available for n ≤ 20.
        </div>
      )}

      {allValid && (
        <div>
          <p
            style={{
              fontSize: 11,
              color: "var(--muted)",
              marginBottom: 12,
              fontFamily: "var(--mono)",
              letterSpacing: "0.05em",
            }}
          >
            COMPUTED MARKET PRICES
          </p>
          <div
            style={{
              background: "var(--card2)",
              borderRadius: 9,
              padding: "10px 14px",
              marginBottom: 10,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span
              style={{
                fontSize: 11,
                fontFamily: "var(--mono)",
                color: "var(--muted)",
                letterSpacing: "0.08em",
              }}
            >
              Y — NONE HAPPEN
            </span>
            <span
              style={{
                fontSize: 18,
                fontFamily: "var(--mono)",
                fontWeight: 700,
                color: "var(--fg)",
              }}
            >
              {priceY.toFixed(6)}
            </span>
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 6,
              maxHeight: 360,
              overflowY: "auto",
              paddingRight: 4,
            }}
          >
            {prices.map((price, i) => (
              <ResultBar
                key={i}
                label={`E${i + 1}`}
                value={price}
                color={COLORS[i % COLORS.length]}
              />
            ))}
          </div>
          <div
            style={{
              marginTop: 10,
              padding: "8px 14px",
              background: "var(--card2)",
              borderRadius: 7,
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <span style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--muted)" }}>
              Σ all prices
            </span>
            <span
              style={{
                fontSize: 12,
                fontFamily: "var(--mono)",
                fontWeight: 600,
                color: "var(--fg)",
              }}
            >
              {(priceY + prices.reduce((a, b) => a + b, 0)).toFixed(6)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [mode, setMode] = useState<"pricesToProbs" | "probsToPrices">("pricesToProbs");
  const [n, setN] = useState(3);
  const [lr, setLr] = useState(0.01);
  const [iters, setIters] = useState(3000);
  const [showSettings, setShowSettings] = useState(false);
  const { priceY, prices } = computePrices([0.382, 0.655, 0.531]);
  console.log(prices);
  const state = useImpliedProbsAsync(priceY, prices, true);
  console.log(state);
  return (
    <>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
          --bg: #0d0d0f; --card: #16161a; --card2: #1e1e24; --border: #2a2a32;
          --fg: #e8e8f0; --muted: #5a5a72; --accent: #6366f1; --accent2: #06b6d4;
          --input-bg: #121216; --input-disabled: #0f0f12; --track: #1e1e26;
          --error: #ef4444; --warn: #f59e0b; --warn-bg: rgba(245,158,11,0.08);
          --success: #10b981; --success-bg: rgba(16,185,129,0.08);
          --mono: 'JetBrains Mono', 'Fira Code', monospace;
          --display: 'Syne', sans-serif;
        }
        html, body, #root { min-height: 100vh; background: var(--bg); color: var(--fg); }
        body { font-family: var(--display); -webkit-font-smoothing: antialiased; }
        input[type=number] { -moz-appearance: textfield; appearance: textfield; }
        input[type=number]::-webkit-outer-spin-button,
        input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; }
        textarea { outline: none; font-family: var(--mono); }
        .tab-btn { flex: 1; padding: 10px 0; background: transparent; border: none; border-radius: 8px; font-family: var(--mono); font-size: 12px; font-weight: 600; letter-spacing: 0.08em; cursor: pointer; transition: background 0.15s, color 0.15s; color: var(--muted); }
        .tab-btn.active { background: var(--accent); color: #fff; }
        .tab-btn:hover:not(.active) { background: var(--card2); color: var(--fg); }
        .n-btn { min-width: 36px; height: 34px; padding: 0 8px; border-radius: 6px; border: 1.5px solid var(--border); background: transparent; color: var(--fg); font-family: var(--mono); font-size: 12px; font-weight: 700; cursor: pointer; transition: background 0.12s, border-color 0.12s; }
        .n-btn.selected { background: var(--card2); border-color: var(--accent); color: var(--accent); }
        .n-btn:hover:not(.selected) { background: var(--card2); }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
      `}</style>

      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "center",
          padding: "40px 16px 80px",
        }}
      >
        <div style={{ width: "100%", maxWidth: 520 }}>
          <div style={{ marginBottom: 28 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <div
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: "50%",
                  background: "var(--accent)",
                  boxShadow: "0 0 10px var(--accent)",
                }}
              />
              <span
                style={{
                  fontSize: 10,
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
                fontSize: 30,
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
            <p style={{ marginTop: 8, fontSize: 13, color: "var(--muted)", lineHeight: 1.6 }}>
              Analytical gradient · Adam optimizer · Web Worker
            </p>
          </div>

          <div
            style={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 16,
              padding: 20,
              display: "flex",
              flexDirection: "column",
              gap: 18,
            }}
          >
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

            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-end",
                gap: 12,
              }}
            >
              <div>
                <p
                  style={{
                    fontSize: 10,
                    fontFamily: "var(--mono)",
                    color: "var(--muted)",
                    letterSpacing: "0.1em",
                    marginBottom: 8,
                  }}
                >
                  NUMBER OF EVENTS
                </p>
                <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
                  {[3, 5, 10, 20, 50, 100].map((v) => (
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
              <button
                onClick={() => setShowSettings((s) => !s)}
                style={{
                  background: "transparent",
                  border: "1px solid var(--border)",
                  borderRadius: 7,
                  color: showSettings ? "var(--accent)" : "var(--muted)",
                  fontSize: 10,
                  fontFamily: "var(--mono)",
                  padding: "6px 10px",
                  cursor: "pointer",
                  letterSpacing: "0.08em",
                  flexShrink: 0,
                }}
              >
                ⚙ SETTINGS
              </button>
            </div>

            {showSettings && (
              <div
                style={{
                  background: "var(--card2)",
                  borderRadius: 10,
                  padding: 14,
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}
              >
                <p
                  style={{
                    fontSize: 10,
                    fontFamily: "var(--mono)",
                    color: "var(--muted)",
                    letterSpacing: "0.1em",
                  }}
                >
                  ADAM OPTIMIZER SETTINGS
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {(
                    [
                      [
                        "LEARNING RATE",
                        lr,
                        0.001,
                        0.0001,
                        0.5,
                        (v: string) => setLr(parseFloat(v)),
                      ],
                      ["ITERATIONS", iters, 100, 100, 20000, (v: string) => setIters(parseInt(v))],
                    ] as const
                  ).map(([lbl, val, step, min, max, set]) => (
                    <div
                      key={lbl as string}
                      style={{ display: "flex", flexDirection: "column", gap: 5 }}
                    >
                      <label
                        style={{
                          fontSize: 10,
                          fontFamily: "var(--mono)",
                          color: "var(--muted)",
                          letterSpacing: "0.08em",
                        }}
                      >
                        {lbl as string}
                      </label>
                      <input
                        type="number"
                        step={step as number}
                        min={min as number}
                        max={max as number}
                        value={val as number}
                        onChange={(e) => (set as Function)(e.target.value)}
                        style={{
                          padding: "7px 10px",
                          background: "var(--input-bg)",
                          border: "1.5px solid var(--border)",
                          borderRadius: 7,
                          color: "var(--fg)",
                          fontSize: 13,
                          fontFamily: "var(--mono)",
                          outline: "none",
                        }}
                      />
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--muted)" }}>
                  Default lr=0.01 works well for any n. Increase iters for n=100.
                </p>
              </div>
            )}

            <div style={{ height: 1, background: "var(--border)" }} />

            {mode === "pricesToProbs" ? (
              <PricesToProbs key={`p2p-${n}`} n={n} lr={lr} iters={iters} />
            ) : (
              <ProbsToPrices key={`pr2p-${n}`} n={n} />
            )}
          </div>

          <p
            style={{
              marginTop: 14,
              textAlign: "center",
              fontSize: 10,
              fontFamily: "var(--mono)",
              color: "var(--muted)",
              letterSpacing: "0.05em",
            }}
          >
            analytical gradient · adam · web worker · ui never freezes
          </p>
        </div>
      </div>
    </>
  );
}
