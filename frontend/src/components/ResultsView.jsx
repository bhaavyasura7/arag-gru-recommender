const NAMES = {
  random: "Random",
  popularity: "Popularity",
  gru_seed0: "GRU (seed 0)",
  gru_seed1: "GRU (seed 1)",
  gru_seed2: "GRU (seed 2)",
  recency: "LLM-Recency",
  vanilla_rag: "LLM-Vanilla-RAG",
  arag: "ARAG",
  arag_wo_nli: "ARAG w/o NLI",
  arag_wo_nli_csa: "ARAG w/o NLI & CSA",
  arag_gru: "ARAG + GRU",
}
const ORDER = [
  "random", "popularity", "gru_seed0", "gru_seed1", "gru_seed2",
  "recency", "vanilla_rag", "arag", "arag_wo_nli", "arag_wo_nli_csa", "arag_gru",
]
const METRICS = ["hit@5", "ndcg@5", "hit@10", "ndcg@10", "mrr"]

function AlphaCurve({ val, test, best }) {
  const alphas = Object.keys(val).map(Number).sort((a, b) => a - b)
  if (alphas.length === 0) return null
  const w = 640, h = 280, padL = 48, padB = 40, padT = 16, padR = 16
  const x = (a) => padL + a * (w - padL - padR)
  const y = (v) => h - padB - v * (h - padB - padT)
  const path = (obj) =>
    alphas.map((a, i) => `${i ? "L" : "M"}${x(a).toFixed(1)},${y(obj[a]).toFixed(1)}`).join(" ")
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart" role="img" aria-label="alpha curve">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={padL} y1={y(g)} x2={w - padR} y2={y(g)} className="c-grid" />
          <text x={padL - 8} y={y(g) + 4} textAnchor="end" className="c-label">{g.toFixed(2)}</text>
        </g>
      ))}
      <line x1={padL} y1={padT} x2={padL} y2={h - padB} className="c-axis" />
      <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} className="c-axis" />
      {alphas.map((a) => (
        <text key={a} x={x(a)} y={h - padB + 16} textAnchor="middle" className="c-label">
          {a.toFixed(1)}
        </text>
      ))}
      <line x1={x(best)} y1={padT} x2={x(best)} y2={h - padB} className="c-best" />
      <path d={path(val)} className="c-line c-line-val" />
      <path d={path(test)} className="c-line c-line-test" />
      <text x={w - padR} y={padT + 12} textAnchor="end" className="c-label">NDCG@5</text>
      <text x={(padL + w) / 2} y={h - 6} textAnchor="middle" className="c-label">
        α  (0 = GRU only, 1 = ARAG only)
      </text>
    </svg>
  )
}

function sig(ci) {
  return ci && (ci[0] > 0 || ci[1] < 0)
}

export default function ResultsView({ data }) {
  if (!data) return <p className="muted">Loading benchmarks…</p>
  const m = data.metrics || {}
  const delta = (sys, ref) => m[`${sys}_vs_${ref}`]?.["ndcg@5"]
  const h1 = delta("arag_gru", "arag")
  const h2 = delta("arag_gru", "gru_seed0")
  const val = Object.fromEntries(Object.entries(data.alpha_curve_validation).map(([k, v]) => [Number(k), v]))
  const test = Object.fromEntries(Object.entries(data.alpha_curve_test).map(([k, v]) => [Number(k), v]))
  const gruSeeds = ["gru_seed0", "gru_seed1", "gru_seed2"].map((s) => m[s]?.["ndcg@5"]?.mean).filter((x) => x != null)
  const gruMean = gruSeeds.length ? gruSeeds.reduce((a, b) => a + b, 0) / gruSeeds.length : null

  return (
    <div className="results">
      <div className="page-head">
        <h2>Benchmarks</h2>
        <p className="muted">
          Amazon Reviews 2023 · Video_Games · frozen 20-item candidate lists (target + 19 uniform negatives) ·
          100 test users for LLM systems, all 94,762 test users for non-LLM systems · paired-bootstrap 95% CIs
          (10,000 resamples).
        </p>
      </div>

      <div className="stat-row">
        <div className="stat"><div className="stat-k">Dataset</div><div className="stat-v small">Video_Games</div></div>
        <div className="stat"><div className="stat-k">Test users</div><div className="stat-v">100</div></div>
        <div className="stat"><div className="stat-k">Best θ</div><div className="stat-v">{data.best_theta}</div></div>
        <div className="stat"><div className="stat-k">Best α</div><div className="stat-v">{data.best_alpha}</div></div>
        <div className="stat"><div className="stat-k">LLM cost</div><div className="stat-v">${(data.cost?.cost_usd ?? 0).toFixed(3)}</div></div>
        <div className="stat"><div className="stat-k">Model</div><div className="stat-v small">{data.model || "gpt-4o-mini"}</div></div>
      </div>

      <div className="finding">
        <div className={`finding-item ${sig(h1?.ci) ? "sig" : ""}`}>
          <div className="finding-k">H1 · ARAG + GRU vs ARAG alone</div>
          <div className="finding-v">ΔNDCG@5 {h1 ? `${h1.mean_diff >= 0 ? "+" : ""}${h1.mean_diff.toFixed(3)}` : "—"}</div>
          <div className="finding-ci">{h1 ? `95% CI [${h1.ci[0].toFixed(3)}, ${h1.ci[1].toFixed(3)}] · ${sig(h1.ci) ? "significant" : "no significant difference"}` : ""}</div>
        </div>
        <div className={`finding-item ${sig(h2?.ci) ? "sig" : ""}`}>
          <div className="finding-k">H2 · ARAG + GRU vs GRU alone</div>
          <div className="finding-v">ΔNDCG@5 {h2 ? `${h2.mean_diff >= 0 ? "+" : ""}${h2.mean_diff.toFixed(3)}` : "—"}</div>
          <div className="finding-ci">{h2 ? `95% CI [${h2.ci[0].toFixed(3)}, ${h2.ci[1].toFixed(3)}] · ${sig(h2.ci) ? "significant" : "no significant difference"}` : ""}</div>
        </div>
      </div>

      <h3>All systems (test set)</h3>
      <div className="table-wrap">
        <table className="metrics">
          <thead>
            <tr>
              <th>System</th>
              {METRICS.map((k) => <th key={k}>{k}</th>)}
              <th>vs ARAG (NDCG@5)</th>
            </tr>
          </thead>
          <tbody>
            {ORDER.filter((s) => m[s]).map((s) => {
              const d = delta(s, "arag")
              return (
                <tr key={s} className={s === "arag_gru" ? "hl" : ""}>
                  <td>{NAMES[s] || s}</td>
                  {METRICS.map((k) => <td key={k}>{(m[s][k]?.mean ?? 0).toFixed(3)}</td>)}
                  <td className={sig(d?.ci) ? "sig-cell" : ""}>
                    {d ? `${d.mean_diff >= 0 ? "+" : ""}${d.mean_diff.toFixed(3)} ${sig(d.ci) ? "✓" : "–"}` : "—"}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="muted">
        “vs ARAG” = mean per-user NDCG@5 difference; ✓ = 95% CI excludes 0 (significant), – = includes 0.
        GRU alone is the mean of 3 seeds (NDCG@5 {gruMean ? gruMean.toFixed(3) : "—"}).
      </p>

      <h3>Fusion weight α — validation vs test NDCG@5</h3>
      <AlphaCurve val={val} test={test} best={data.best_alpha} />
      <div className="legend">
        <span className="legend-item"><span className="swatch swatch-val" /> validation</span>
        <span className="legend-item"><span className="swatch swatch-test" /> test</span>
        <span className="legend-item"><span className="swatch swatch-best" /> α* = {data.best_alpha}</span>
      </div>

      <h3>Protocol &amp; reproduction</h3>
      <ul className="protocol">
        <li>θ ∈ {"{4,5,6,7}"} and α ∈ [0,1] tuned on <b>validation</b> NDCG@5; test α is never tuned on test.</li>
        <li>All systems rank the same frozen lists; α=1 reproduces IRA order, α=0 reproduces GRU order.</li>
        <li>Every number here is produced by the pipeline and audited by <code>make verify</code>; re-running the eval reuses the LLM cache (zero new API calls).</li>
        <li>Full write-up: <code>results/report.md</code> · metrics: <code>results/metrics.json</code> · process: <code>docs/PROCESS.md</code>.</li>
      </ul>
    </div>
  )
}
