const NAMES = {
  random: "Random",
  popularity: "Popularity",
  content: "Content similarity",
  itemknn: "ItemKNN",
  gru: "GRU",
  gru_text: "Text-aware GRU",
  sasrec: "SASRec",
  recency: "LLM-Recency",
  vanilla_rag: "LLM-Vanilla-RAG",
  arag: "ARAG",
  arag_gru: "ARAG + GRU",
  arag_gru_text: "ARAG + text-GRU",
  arag_content: "ARAG + content",
  arag_wo_nli: "ARAG w/o NLI",
  arag_wo_nli_csa: "ARAG w/o NLI & CSA",
}
const FAMILY = {
  random: "baseline", popularity: "baseline", content: "classic", itemknn: "classic",
  gru: "trained", gru_text: "trained", sasrec: "trained",
  recency: "llm", vanilla_rag: "llm", arag: "llm",
  arag_gru: "hybrid", arag_gru_text: "hybrid", arag_content: "hybrid",
}
const COLS = [["hit@5", "Hit@5"], ["ndcg@10", "NDCG@10"], ["hit@10", "Hit@10"], ["mrr", "MRR"]]
const FAMILY_LABEL = { baseline: "Baseline", classic: "Classic", trained: "Trained model", llm: "LLM", hybrid: "ARAG + model" }
const TRAINED = ["gru", "gru_text", "sasrec", "content"]
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

const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString())
const f3 = (x) => (x == null ? "—" : x.toFixed(3))
const fd = (d) => (d == null ? "—" : `${d >= 0 ? "+" : ""}${d.toFixed(3)}`)
const flip = (t) => t && { mean_diff: -t.mean_diff, ci: [-t.ci[1], -t.ci[0]] }

function Finding({ k, test, note }) {
  return (
    <div className={`finding-item ${sig(test?.ci) ? (test.mean_diff > 0 ? "sig" : "neg") : ""}`}>
      <div className="finding-k">{k}</div>
      <div className="finding-v">ΔNDCG@5 {fd(test?.mean_diff)}</div>
      <div className="finding-ci">
        {test ? `95% CI [${f3(test.ci[0])}, ${f3(test.ci[1])}] · ${sig(test.ci) ? "significant" : "not significant"}` : ""}
      </div>
      {note && <div className="finding-note">{note}</div>}
    </div>
  )
}

export default function ResultsView({ data }) {
  if (!data) return <p className="muted">Loading benchmarks…</p>
  const m = data.metrics || {}
  const ex = data.extra || {}
  const st = ex.sasrec_tests || {}
  const h1 = m.arag_gru_vs_arag?.["ndcg@5"]
  const h2 = m.arag_gru_vs_gru_seed0?.["ndcg@5"]
  const vsSasrec = st["ARAG vs SASRec"]
  const beyond = ex.beyond_accuracy_100 || {}

  // paired difference vs ARAG on the 100 users (3-seed averages for the trained models)
  const vsArag = (s) =>
    ({ gru: flip(st["ARAG vs GRU (ID)"]), gru_text: flip(st["ARAG vs Text-aware GRU"]), sasrec: flip(st["ARAG vs SASRec"]) })[s] ??
    m[`${s}_vs_arag`]?.["ndcg@5"]

  const overall = { ...(ex.overall_100 || {}) }
  if (m.arag_content && !overall.arag_content) overall.arag_content = m.arag_content
  const board = Object.keys(overall)
    .filter((s) => NAMES[s])
    .sort((a, b) => overall[b]["ndcg@5"].mean - overall[a]["ndcg@5"].mean)
  const top = board.length ? overall[board[0]]["ndcg@5"].mean : 1
  const all = ex.overall_all || {}
  const allRows = Object.keys(all).sort((a, b) => all[b]["ndcg@5"].mean - all[a]["ndcg@5"].mean)
  const hist = ex.history_all || {}
  const trainedPop = ["gru", "gru_text", "sasrec"].map((s) => beyond[s]?.popularity_lift?.mean).filter((x) => x != null)
  const trainedPopMean = trainedPop.length ? trainedPop.reduce((a, b) => a + b, 0) / trainedPop.length : null

  const val = Object.fromEntries(Object.entries(data.alpha_curve_validation).map(([k, v]) => [Number(k), v]))
  const test = Object.fromEntries(Object.entries(data.alpha_curve_test).map(([k, v]) => [Number(k), v]))

  return (
    <div className="results">
      <div className="page-head">
        <h2>Benchmarks</h2>
        <p className="muted">
          Every system ranks the same 20 candidate games per customer (the real next game + 19 random ones). Score =
          NDCG@5: 1.0 means the real game was ranked first. LLM systems are tested on 100 customers (API cost); the
          others on all 94,762. Paired-bootstrap 95% confidence intervals.
        </p>
      </div>

      <h3>Key findings</h3>
      <div className="finding">
        <Finding k="H1 · Does adding a GRU help ARAG?" test={h1} note="Yes — the blend clearly beats ARAG alone." />
        <Finding k="H2 · Does the blend beat the GRU alone?" test={h2} note="No — about the same as the GRU." />
        <Finding k="ARAG vs SASRec (best trained model)" test={vsSasrec} note="Trained sequence models are more accurate than ARAG alone." />
        <div className={`finding-item ${beyond.arag ? "sig" : ""}`}>
          <div className="finding-k">Popularity bias (lower = less "just popular games")</div>
          <div className="finding-v">ARAG {f3(beyond.arag?.popularity_lift?.mean)} vs trained {f3(trainedPopMean)}</div>
          <div className="finding-note">ARAG's strength: it recommends beyond the bestsellers, and explains every pick.</div>
        </div>
      </div>

      {board.length > 0 && (
        <>
          <h3>Leaderboard · same 100 test customers</h3>
          <div className="table-wrap">
            <table className="metrics">
              <thead>
                <tr><th>System</th><th>Type</th><th className="bar-col">NDCG@5</th>{COLS.map(([k, l]) => <th key={k}>{l}</th>)}<th>vs ARAG</th></tr>
              </thead>
              <tbody>
                {board.map((s) => {
                  const r = overall[s]
                  const d = vsArag(s)
                  return (
                    <tr key={s} className={s === "arag" ? "hl" : ""}>
                      <td>{NAMES[s]}</td>
                      <td><span className={`fam fam-${FAMILY[s]}`}>{FAMILY_LABEL[FAMILY[s]]}</span></td>
                      <td className="bar-col">
                        <div className="bar"><span style={{ width: `${(100 * r["ndcg@5"].mean) / top}%` }} /></div>
                        <b>{f3(r["ndcg@5"].mean)}</b>
                      </td>
                      {COLS.map(([k]) => <td key={k}>{f3(r[k]?.mean)}</td>)}
                      <td className={sig(d?.ci) ? "sig-cell" : ""}>{s === "arag" ? "—" : `${fd(d?.mean_diff)} ${sig(d?.ci) ? "✓" : "–"}`}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="muted small-note">
            ✓ = significant difference from ARAG (95% CI excludes 0). GRU, text-GRU and SASRec are 3-seed averages. With
            100 customers, gaps under ~0.03 are not reliable. NDCG@k / Hit@k look at the top k of 20; MRR = 1 / rank of the real game.
          </p>
        </>
      )}

      {allRows.length > 0 && (
        <>
          <h3>All {fmt(data.n_test_users_all)} test customers · non-LLM models</h3>
          <div className="table-wrap">
            <table className="metrics">
              <thead><tr><th>System</th><th>NDCG@5</th><th>95% CI</th>{COLS.map(([k, l]) => <th key={k}>{l}</th>)}</tr></thead>
              <tbody>
                {allRows.map((s) => (
                  <tr key={s} className={s === allRows[0] ? "hl" : ""}>
                    <td>{NAMES[s] || s}</td>
                    <td>{f3(all[s]["ndcg@5"].mean)}</td>
                    <td>[{f3(all[s]["ndcg@5"].ci[0])}, {f3(all[s]["ndcg@5"].ci[1])}]</td>
                    {COLS.map(([k]) => <td key={k}>{f3(all[s][k]?.mean)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small-note">On every customer, reading game descriptions (text-aware GRU) is the best single model.</p>
        </>
      )}

      {Object.keys(beyond).length > 0 && (
        <>
          <h3>Beyond accuracy · 100 test customers</h3>
          <div className="table-wrap">
            <table className="metrics">
              <thead><tr><th>System</th><th>Popularity bias ↓</th><th>Diversity ↑</th><th>Catalogue coverage ↑</th></tr></thead>
              <tbody>
                {Object.keys(beyond).filter((s) => NAMES[s]).map((s) => (
                  <tr key={s} className={s === "arag" ? "hl" : ""}>
                    <td>{NAMES[s]}</td>
                    <td>{f3(beyond[s].popularity_lift?.mean)}</td>
                    <td>{f3(beyond[s].diversity_ild?.mean)}</td>
                    <td>{f3(beyond[s].coverage)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small-note">
            Popularity bias = how much more popular the top-5 picks are than the candidates on average (0 = no bias).
            Diversity = how different the top-5 games are from each other. Coverage = share of games ever recommended.
          </p>
        </>
      )}

      {Object.keys(hist).length > 0 && (
        <>
          <h3>Does history length matter? · all test customers</h3>
          <div className="table-wrap">
            <table className="metrics">
              <thead>
                <tr><th>Games in history</th><th>Customers</th>{TRAINED.map((s) => <th key={s}>{NAMES[s]}</th>)}</tr>
              </thead>
              <tbody>
                {Object.entries(hist).map(([bin, row]) => (
                  <tr key={bin}>
                    <td>{bin}</td>
                    <td>{fmt(row.n_users)}</td>
                    {TRAINED.map((s) => <td key={s}>{f3(row[s]?.mean)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small-note">Almost flat: customers with only 4 games get nearly the same quality as those with 11+.</p>
        </>
      )}

      <h3>Blend weight α · validation vs test (best α = {data.best_alpha})</h3>
      <p className="muted small-note">Final score = α·ARAG + (1−α)·GRU. α is picked on validation only, never on test.</p>
      <AlphaCurve val={val} test={test} best={data.best_alpha} />
      <div className="legend">
        <span className="legend-item"><span className="swatch swatch-val" /> validation</span>
        <span className="legend-item"><span className="swatch swatch-test" /> test</span>
        <span className="legend-item"><span className="swatch swatch-best" /> α* = {data.best_alpha}</span>
      </div>

      {m.arag_wo_nli && (
        <details className="more">
          <summary>ARAG ablations (removing agents)</summary>
          <div className="table-wrap">
            <table className="metrics">
              <thead><tr><th>System</th><th>NDCG@5</th>{COLS.map(([k, l]) => <th key={k}>{l}</th>)}<th>vs ARAG</th></tr></thead>
              <tbody>
                {["arag", "arag_wo_nli", "arag_wo_nli_csa"].map((s) => {
                  const d = m[`${s}_vs_arag`]?.["ndcg@5"]
                  return (
                    <tr key={s}>
                      <td>{NAMES[s]}</td>
                      <td>{f3(m[s]?.["ndcg@5"]?.mean)}</td>
                      {COLS.map(([k]) => <td key={k}>{f3(m[s]?.[k]?.mean)}</td>)}
                      <td className={sig(d?.ci) ? "sig-cell" : ""}>{s === "arag" ? "—" : `${fd(d?.mean_diff)} ${sig(d?.ci) ? "✓" : "–"}`}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </details>
      )}

      <h3>Protocol &amp; reproduction</h3>
      <ul className="protocol">
        <li>θ and α are tuned on <b>validation</b> NDCG@5 only; nothing is tuned on test.</li>
        <li>All systems rank the same frozen candidate lists; α=1 is pure ARAG order, α=0 pure GRU order.</li>
        <li>Every number is produced by the pipeline and checked by <code>make verify</code>; re-running reuses the saved LLM answers (no new API calls).</li>
        <li>Write-up: <code>docs/REPORT.md</code> · metrics: <code>results/metrics.json</code>, <code>results/extra_metrics.json</code>.</li>
      </ul>
    </div>
  )
}
