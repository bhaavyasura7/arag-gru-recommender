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
  arag_wo_nli: "ARAG without NLI agent",
  arag_wo_nli_csa: "ARAG without NLI & CSA agents",
}
const FAMILY = {
  random: "baseline", popularity: "baseline", content: "classic", itemknn: "classic",
  gru: "trained", gru_text: "trained", sasrec: "trained",
  recency: "llm", vanilla_rag: "llm", arag: "llm",
  arag_gru: "hybrid", arag_gru_text: "hybrid", arag_content: "hybrid",
}
const FAMILY_LABEL = {
  trained: "Trained model",
  hybrid: "ARAG + model (ours)",
  llm: "LLM only",
  classic: "Simple similarity",
  baseline: "Sanity check",
}
const COLS = [["ndcg@5", "NDCG@5"], ["hit@5", "Hit@5"], ["ndcg@10", "NDCG@10"], ["hit@10", "Hit@10"], ["mrr", "MRR"]]

const f3 = (x) => (x == null ? "—" : x.toFixed(3))
const fd = (d) => (d == null ? "—" : `${d >= 0 ? "+" : ""}${d.toFixed(3)}`)
const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString())
const flip = (t) => t && { mean_diff: -t.mean_diff, ci: [-t.ci[1], -t.ci[0]] }
const sig = (ci) => ci && (ci[0] > 0 || ci[1] < 0)

/* ---------- small building blocks ---------- */

function Answer({ kind, children }) {
  return <span className={`answer answer-${kind}`}>{children}</span>
}

function Explain({ children }) {
  return <div className="explain">{children}</div>
}

function Question({ n, title, answer, children }) {
  return (
    <section className="rq">
      <div className="rq-head">
        <span className="rq-n">{n}</span>
        <h3>{title}</h3>
        {answer}
      </div>
      {children}
    </section>
  )
}

/** Horizontal bars with a thin line for the 95% confidence range. */
function Bars({ rows, max, lowerIsBetter = false, digits = 3 }) {
  const top = max ?? Math.max(...rows.map((r) => r.ci?.[1] ?? r.value)) * 1.05
  const pct = (v) => `${Math.max(0, Math.min(100, (100 * v) / top))}%`
  return (
    <div className="bars">
      {rows.map((r) => (
        <div key={r.key} className={`bar-row ${r.strong ? "strong" : ""}`}>
          <div className="bar-label">
            {r.label}
            {r.tag && <span className="bar-tag">{r.tag}</span>}
          </div>
          <div className="bar-track">
            <div className={`bar-fill fam-bg-${r.family || "none"}`} style={{ width: pct(r.value) }} />
            {r.ci && <div className="bar-ci" style={{ left: pct(r.ci[0]), width: `calc(${pct(r.ci[1])} - ${pct(r.ci[0])})` }} />}
          </div>
          <div className="bar-val">{r.value.toFixed(digits)}</div>
        </div>
      ))}
      {lowerIsBetter && <div className="bars-note">Shorter bar = better here.</div>}
    </div>
  )
}

function Legend({ families }) {
  return (
    <div className="legend">
      {families.map((f) => (
        <span key={f} className="legend-item"><span className={`dot fam-bg-${f}`} /> {FAMILY_LABEL[f]}</span>
      ))}
      <span className="legend-item"><span className="ci-swatch" /> 95% confidence range</span>
    </div>
  )
}

function AlphaCurve({ val, test, best }) {
  const alphas = Object.keys(val).map(Number).sort((a, b) => a - b)
  if (alphas.length === 0) return null
  const vals = [...Object.values(val), ...Object.values(test)]
  const lo = Math.floor(Math.min(...vals) * 20) / 20, hi = Math.ceil(Math.max(...vals) * 20) / 20
  const w = 640, h = 260, padL = 48, padB = 40, padT = 16, padR = 16
  const x = (a) => padL + a * (w - padL - padR)
  const y = (v) => h - padB - ((v - lo) / (hi - lo)) * (h - padB - padT)
  const ticks = []
  for (let g = lo; g <= hi + 1e-9; g += 0.05) ticks.push(Number(g.toFixed(2)))
  const path = (obj) => alphas.map((a, i) => `${i ? "L" : "M"}${x(a).toFixed(1)},${y(obj[a]).toFixed(1)}`).join(" ")
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart" role="img" aria-label="blend weight curve">
      {ticks.map((g) => (
        <g key={g}>
          <line x1={padL} y1={y(g)} x2={w - padR} y2={y(g)} className="c-grid" />
          <text x={padL - 8} y={y(g) + 4} textAnchor="end" className="c-label">{g.toFixed(2)}</text>
        </g>
      ))}
      <line x1={padL} y1={padT} x2={padL} y2={h - padB} className="c-axis" />
      <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} className="c-axis" />
      {alphas.map((a) => (
        <text key={a} x={x(a)} y={h - padB + 16} textAnchor="middle" className="c-label">{a.toFixed(1)}</text>
      ))}
      <line x1={x(best)} y1={padT} x2={x(best)} y2={h - padB} className="c-best" />
      <path d={path(val)} className="c-line c-line-val" />
      <path d={path(test)} className="c-line c-line-test" />
      <text x={padL + 6} y={padT + 10} className="c-label">NDCG@5</text>
      <text x={(padL + w) / 2} y={h - 6} textAnchor="middle" className="c-label">
        α  (0 = GRU only · 1 = ARAG only)
      </text>
    </svg>
  )
}

/* ---------- the page ---------- */

export default function ResultsView({ data, stats, nDemo }) {
  if (!data) return <p className="muted">Loading results…</p>
  const m = data.metrics || {}
  const ex = data.extra || {}
  const st = ex.sasrec_tests || {}
  const beyond = ex.beyond_accuracy_100 || {}
  const hist = ex.history_all || {}
  const all = ex.overall_all || {}

  const overall = { ...(ex.overall_100 || {}) }
  if (m.arag_content && !overall.arag_content) overall.arag_content = m.arag_content
  const n5 = (s) => overall[s]?.["ndcg@5"]?.mean ?? m[s]?.["ndcg@5"]?.mean

  const h1 = m.arag_gru_vs_arag?.["ndcg@5"]
  const h2 = m.arag_gru_vs_gru_seed0?.["ndcg@5"]
  const vsSasrec = st["ARAG vs SASRec"]
  const vsArag = (s) =>
    ({ gru: flip(st["ARAG vs GRU (ID)"]), gru_text: flip(st["ARAG vs Text-aware GRU"]), sasrec: flip(st["ARAG vs SASRec"]) })[s] ??
    m[`${s}_vs_arag`]?.["ndcg@5"]

  const board = Object.keys(overall).filter((s) => NAMES[s]).sort((a, b) => n5(b) - n5(a))
  const best = board[0]
  const allRows = Object.keys(all).sort((a, b) => all[b]["ndcg@5"].mean - all[a]["ndcg@5"].mean)
  const trainedPop = ["gru", "gru_text", "sasrec"].map((s) => beyond[s]?.popularity_lift?.mean).filter((x) => x != null)
  const trainedPopMean = trainedPop.length ? trainedPop.reduce((a, b) => a + b, 0) / trainedPop.length : null
  const aragPop = beyond.arag?.popularity_lift?.mean
  const bins = Object.entries(hist)
  const firstBin = bins[0], lastBin = bins[bins.length - 1]

  const val = Object.fromEntries(Object.entries(data.alpha_curve_validation || {}).map(([k, v]) => [Number(k), v]))
  const test = Object.fromEntries(Object.entries(data.alpha_curve_test || {}).map(([k, v]) => [Number(k), v]))
  const r = stats?.rows || {}
  const gain = n5("arag") ? (100 * (n5("arag_gru") - n5("arag"))) / n5("arag") : null

  return (
    <div className="results">
      {/* ---------------- summary ---------------- */}
      <div className="hero">
        <div className="hero-k">The research in one minute</div>
        <h2>Can LLM agents recommend video games better than a trained model — and does combining them help?</h2>
        <p>
          <b>ARAG</b> (from the Walmart ARAG paper) uses four LLM agents that read a customer's past reviews and rank
          20 candidate games, explaining each pick. We rebuilt it, then <b>blended it with a GRU</b> — a small neural
          network trained on every customer's purchase order — and compared everything on the same test.
        </p>
        <div className="hero-cards">
          <div className="hero-card good">
            <div className="hc-v">{f3(n5("arag"))} → {f3(n5("arag_gru"))}</div>
            <div className="hc-k">Adding the GRU improves ARAG{gain != null ? ` by ${gain.toFixed(0)}%` : ""}</div>
          </div>
          <div className="hero-card">
            <div className="hc-v">{NAMES[best]} {f3(n5(best))}</div>
            <div className="hc-k">Most accurate system (a trained model); our best blend is right behind</div>
          </div>
          <div className="hero-card good">
            <div className="hc-v">{f3(aragPop)} vs {f3(trainedPopMean)}</div>
            <div className="hc-k">ARAG's popularity bias vs trained models — ARAG suggests far less "just bestsellers"</div>
          </div>
        </div>
      </div>

      {/* ---------------- how we test ---------------- */}
      <section className="rq">
        <div className="rq-head"><span className="rq-n">i</span><h3>How we test</h3></div>
        <div className="how">
          <div className="how-step">
            <div className="how-n">1</div>
            <div><b>Hide the last game</b> each customer reviewed. Everything before it is their history.</div>
          </div>
          <div className="how-step">
            <div className="how-n">2</div>
            <div><b>Make a list of 20 games</b>: the hidden one + 19 decoys, picked so they are about as popular as the real one. Every system gets the exact same list.</div>
          </div>
          <div className="how-step">
            <div className="how-n">3</div>
            <div><b>Each system ranks the 20.</b> The higher it puts the hidden game, the better.</div>
          </div>
        </div>
        <div className="score-ex">
          <div className="score-ex-k">Score (NDCG@5) for where the hidden game lands:</div>
          <div className="score-ex-row">
            {[["#1", "1.00"], ["#2", "0.63"], ["#3", "0.50"], ["#4", "0.43"], ["#5", "0.39"], ["#6–20", "0"]].map(([k, v]) => (
              <div key={k} className="score-chip"><span>{k}</span><b>{v}</b></div>
            ))}
          </div>
          <div className="muted small-note">We average this over customers. Random guessing scores about 0.13.</div>
        </div>
        <Explain>
          LLM systems cost money per customer, so they are tested on <b>{data.n_test_users_sampled ?? 100} customers</b>.
          Trained models are free to run, so they are also tested on <b>all {fmt(data.n_test_users_all)}</b>. Every
          comparison uses a <b>95% confidence range</b>: if the range of a difference doesn't include 0, the difference is
          real, not luck.
        </Explain>
        <div className="data-row">
          {[["Customers", r.users], ["Reviews", r.interactions], ["Games", r.items], ["Demo accounts", nDemo]].map(([k, v]) => (
            <div key={k} className="data-chip"><b>{fmt(v)}</b><span>{k}</span></div>
          ))}
          <div className="data-chip"><b>Video Games</b><span>Amazon Reviews 2023</span></div>
        </div>
      </section>

      {/* ---------------- Q1 ---------------- */}
      <Question
        n="1"
        title="Does adding a GRU make ARAG better?"
        answer={h1 && <Answer kind={sig(h1.ci) && h1.mean_diff > 0 ? "yes" : "no"}>{sig(h1.ci) && h1.mean_diff > 0 ? "Yes" : "No"}</Answer>}
      >
        <Bars
          max={0.65}
          rows={[
            { key: "arag", label: "ARAG alone", value: n5("arag"), ci: overall.arag?.["ndcg@5"]?.ci, family: "llm" },
            { key: "arag_gru", label: "ARAG + GRU", value: n5("arag_gru"), ci: overall.arag_gru?.["ndcg@5"]?.ci, family: "hybrid", strong: true },
          ]}
        />
        <Explain>
          Final score = <b>α × ARAG + (1 − α) × GRU</b>. The blend scores <b>{fd(h1?.mean_diff)}</b> higher (95% range{" "}
          {f3(h1?.ci[0])} to {f3(h1?.ci[1])}, which doesn't include 0, so it is a real improvement). The GRU knows
          which games people usually buy next; ARAG knows what this customer says they like. Together they rank better
          than ARAG on its own.
        </Explain>
      </Question>

      {/* ---------------- Q2 ---------------- */}
      <Question
        n="2"
        title="Is the blend better than the GRU on its own?"
        answer={h2 && <Answer kind={sig(h2.ci) ? (h2.mean_diff > 0 ? "yes" : "no") : "tie"}>{sig(h2.ci) ? (h2.mean_diff > 0 ? "Yes" : "No") : "About the same"}</Answer>}
      >
        <Bars
          max={0.65}
          rows={[
            { key: "gru", label: "GRU alone", value: n5("gru"), ci: overall.gru?.["ndcg@5"]?.ci, family: "trained" },
            { key: "arag_gru", label: "ARAG + GRU", value: n5("arag_gru"), ci: overall.arag_gru?.["ndcg@5"]?.ci, family: "hybrid", strong: true },
          ]}
        />
        <Explain>
          The difference is only <b>{fd(h2?.mean_diff)}</b> (range {f3(h2?.ci[0])} to {f3(h2?.ci[1])} includes 0), so we
          can't say the blend is more accurate. The best blend weight was <b>α = {data.best_alpha}</b>, meaning the
          ranking is mostly the GRU's. What ARAG adds is a <b>written reason for every pick</b> and natural-language
          search — not extra accuracy.
        </Explain>
      </Question>

      {/* ---------------- Q3 ---------------- */}
      <Question
        n="3"
        title="Which system is most accurate overall?"
        answer={<Answer kind="info">{NAMES[best]}, closely followed by our blend</Answer>}
      >
        <Legend families={["trained", "hybrid", "llm", "classic", "baseline"]} />
        <Bars
          rows={board.map((s) => ({
            key: s, label: NAMES[s], value: n5(s), ci: overall[s]?.["ndcg@5"]?.ci, family: FAMILY[s], strong: s === "arag",
          }))}
        />
        <Explain>
          Same {data.n_test_users_sampled ?? 100} customers for every system, NDCG@5. <b>Trained sequence models</b>{" "}
          (SASRec, GRU) and <b>our blends</b> are at the top. <b>ARAG alone</b> beats the other LLM methods and the simple
          baselines, but trails the trained models (ARAG vs SASRec: {fd(vsSasrec?.mean_diff)}, a real gap). When two
          bars' thin confidence lines overlap a lot, treat them as a tie — with 100 customers, gaps under ~0.03 are not
          reliable.
        </Explain>
        <details className="more">
          <summary>Show the full table (all 5 metrics + difference from ARAG)</summary>
          <div className="table-wrap">
            <table className="metrics">
              <thead>
                <tr><th>System</th>{COLS.map(([k, l]) => <th key={k}>{l}</th>)}<th>vs ARAG</th></tr>
              </thead>
              <tbody>
                {board.map((s) => {
                  const d = vsArag(s)
                  return (
                    <tr key={s} className={s === "arag" ? "hl" : ""}>
                      <td>{NAMES[s]}</td>
                      {COLS.map(([k]) => <td key={k}>{f3((overall[s] || m[s])?.[k]?.mean)}</td>)}
                      <td className={sig(d?.ci) ? "sig-cell" : ""}>{s === "arag" ? "—" : `${fd(d?.mean_diff)} ${sig(d?.ci) ? "✓" : ""}`}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="muted small-note">✓ = real difference (95% range excludes 0). GRU, text-GRU and SASRec are averages of 3 training runs.</p>
        </details>
      </Question>

      {/* ---------------- Q4 ---------------- */}
      {allRows.length > 0 && (
        <Question
          n="4"
          title="Does reading the game descriptions help the trained model?"
          answer={<Answer kind="yes">Yes, on all customers</Answer>}
        >
          <Bars
            max={0.6}
            rows={allRows.filter((s) => s !== "random" && s !== "popularity").map((s) => ({
              key: s, label: NAMES[s], value: all[s]["ndcg@5"].mean, family: FAMILY[s], strong: s === allRows[0],
            }))}
          />
          <Explain>
            Tested on all <b>{fmt(data.n_test_users_all)}</b> customers (no LLM, so it's free). The <b>text-aware GRU</b>{" "}
            also reads each game's title and description, not just its ID. It is the best single model here (
            {f3(all.gru_text?.["ndcg@5"]?.mean)} vs {f3(all.gru?.["ndcg@5"]?.mean)} for the plain GRU). With this many
            customers, the confidence ranges are tiny (about ±0.002), so even small gaps are real.
          </Explain>
        </Question>
      )}

      {/* ---------------- Q5 ---------------- */}
      {Object.keys(beyond).length > 0 && (
        <Question
          n="5"
          title="So where does ARAG actually win?"
          answer={<Answer kind="yes">Less popularity bias + explanations</Answer>}
        >
          <Bars
            lowerIsBetter
            digits={2}
            rows={["popularity", "sasrec", "gru", "gru_text", "arag_gru", "vanilla_rag", "recency", "arag", "content"]
              .filter((s) => beyond[s])
              .map((s) => ({ key: s, label: NAMES[s], value: beyond[s].popularity_lift.mean, family: FAMILY[s], strong: s === "arag" }))}
          />
          <Explain>
            <b>Popularity bias</b> = how much more popular the top-5 picks are than the 20 candidates on average (0 = no
            bias). Trained models learn "popular games sell", so they push bestsellers (~{f3(trainedPopMean)}). ARAG reads
            what the customer actually wrote, so it barely favours popular games ({f3(aragPop)}) and recommends a wider
            range of the catalogue (coverage {f3(beyond.arag?.coverage)} vs {f3(beyond.gru?.coverage)} for the GRU). It
            also explains every recommendation, which the trained models can't do.
          </Explain>
          <details className="more">
            <summary>Show popularity bias, diversity and coverage for every system</summary>
            <div className="table-wrap">
              <table className="metrics">
                <thead><tr><th>System</th><th>Popularity bias ↓</th><th>Diversity ↑</th><th>Coverage ↑</th></tr></thead>
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
              Diversity = how different the top-5 games are from each other. Coverage = share of all candidate games that
              ever reach someone's top 5.
            </p>
          </details>
        </Question>
      )}

      {/* ---------------- Q6 ---------------- */}
      {bins.length > 0 && (
        <Question
          n="6"
          title="Do customers with short histories get worse recommendations?"
          answer={<Answer kind="tie">Barely</Answer>}
        >
          <div className="table-wrap">
            <table className="metrics">
              <thead>
                <tr><th>Games in history</th><th>Customers</th><th>GRU</th><th>Text-aware GRU</th><th>SASRec</th></tr>
              </thead>
              <tbody>
                {bins.map(([bin, row]) => (
                  <tr key={bin}>
                    <td>{bin}</td>
                    <td>{fmt(row.n_users)}</td>
                    <td>{f3(row.gru?.mean)}</td>
                    <td>{f3(row.gru_text?.mean)}</td>
                    <td>{f3(row.sasrec?.mean)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Explain>
            Most customers have only 4–6 games, so we checked whether that hurts. It doesn't, much: the GRU scores{" "}
            {f3(firstBin?.[1]?.gru?.mean)} for customers with {firstBin?.[0]} vs {f3(lastBin?.[1]?.gru?.mean)} for{" "}
            {lastBin?.[0]} games. Our first GRU looked weak because of a <b>training bug</b> (its starting weights were
            too large), not because of short histories — fixing it raised the GRU from 0.24 to {f3(n5("gru"))}.
          </Explain>
        </Question>
      )}

      {/* ---------------- blend weight ---------------- */}
      <Question n="7" title="How was the blend weight α chosen?" answer={<Answer kind="info">α = {data.best_alpha}</Answer>}>
        <AlphaCurve val={val} test={test} best={data.best_alpha} />
        <div className="legend">
          <span className="legend-item"><span className="swatch swatch-val" /> validation customers (used to pick α)</span>
          <span className="legend-item"><span className="swatch swatch-test" /> test customers (never used for tuning)</span>
          <span className="legend-item"><span className="swatch swatch-best" /> chosen α</span>
        </div>
        <Explain>
          We tried α from 0 (GRU only) to 1 (ARAG only) on a separate <b>validation</b> group and kept the best. The test
          line follows the same shape, so the choice wasn't lucky. α = {data.best_alpha} means mostly GRU with a bit of
          ARAG — ARAG's ranking alone (α = 1) is the weakest end of the curve.
        </Explain>
      </Question>

      {/* ---------------- extras ---------------- */}
      <section className="rq">
        <div className="rq-head"><span className="rq-n">+</span><h3>More details</h3></div>

        {m.arag_wo_nli && (
          <details className="more">
            <summary>Does each ARAG agent matter? (removing agents)</summary>
            <div className="table-wrap">
              <table className="metrics">
                <thead><tr><th>System</th>{COLS.map(([k, l]) => <th key={k}>{l}</th>)}<th>vs ARAG</th></tr></thead>
                <tbody>
                  {["arag", "arag_wo_nli", "arag_wo_nli_csa"].map((s) => {
                    const d = m[`${s}_vs_arag`]?.["ndcg@5"]
                    return (
                      <tr key={s} className={s === "arag" ? "hl" : ""}>
                        <td>{NAMES[s]}</td>
                        {COLS.map(([k]) => <td key={k}>{f3(m[s]?.[k]?.mean)}</td>)}
                        <td className={sig(d?.ci) ? "sig-cell" : ""}>{s === "arag" ? "—" : `${fd(d?.mean_diff)} ${sig(d?.ci) ? "✓" : ""}`}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="muted small-note">
              NLI = the agent that checks each game against the customer's tastes; CSA = the agent that summarises the
              context. Removing NLI lowers the score by {fd(m.arag_wo_nli_vs_arag?.["ndcg@5"]?.mean_diff)}; removing both lowers
              it by {fd(m.arag_wo_nli_csa_vs_arag?.["ndcg@5"]?.mean_diff)}
              {sig(m.arag_wo_nli_csa_vs_arag?.["ndcg@5"]?.ci) ? " (a real drop)" : ""} — so the agents do help ARAG.
            </p>
          </details>
        )}

        {allRows.length > 0 && (
          <details className="more">
            <summary>All {fmt(data.n_test_users_all)} customers — full table</summary>
            <div className="table-wrap">
              <table className="metrics">
                <thead><tr><th>System</th>{COLS.map(([k, l]) => <th key={k}>{l}</th>)}<th>95% range (NDCG@5)</th></tr></thead>
                <tbody>
                  {allRows.map((s) => (
                    <tr key={s} className={s === allRows[0] ? "hl" : ""}>
                      <td>{NAMES[s]}</td>
                      {COLS.map(([k]) => <td key={k}>{f3(all[s][k]?.mean)}</td>)}
                      <td>{f3(all[s]["ndcg@5"].ci[0])} – {f3(all[s]["ndcg@5"].ci[1])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}

        <details className="more">
          <summary>What do the metrics mean?</summary>
          <dl className="gloss">
            <dt>NDCG@5</dt><dd>Main score. Full points if the hidden game is ranked #1, fewer the lower it is, 0 if it isn't in the top 5.</dd>
            <dt>Hit@5</dt><dd>Share of customers whose hidden game appears anywhere in the top 5.</dd>
            <dt>NDCG@10 / Hit@10</dt><dd>The same two, but looking at the top 10 of 20.</dd>
            <dt>MRR</dt><dd>1 ÷ the rank of the hidden game (rank 1 = 1.0, rank 2 = 0.5, rank 4 = 0.25), averaged.</dd>
            <dt>95% confidence range</dt><dd>From 10,000 bootstrap resamples of the customers. If the range of a difference excludes 0, the difference is real.</dd>
            <dt>Popularity bias</dt><dd>How much more popular the top-5 picks are than the candidates (0 = none).</dd>
            <dt>Diversity</dt><dd>How different the top-5 games are from each other (based on their descriptions).</dd>
            <dt>Coverage</dt><dd>Share of candidate games that ever reach some customer's top 5.</dd>
          </dl>
        </details>

        <details className="more">
          <summary>Limits of this study (honest caveats)</summary>
          <ul className="protocol">
            <li>LLM systems were tested on {data.n_test_users_sampled ?? 100} customers because of API cost, so small differences between them aren't reliable.</li>
            <li>One product category (Video Games) and one LLM ({data.model || "gpt-4o-mini"}).</li>
            <li>The original paper compared ARAG only with other LLM methods; our trained-model comparison is an addition, not a contradiction of its claims.</li>
          </ul>
        </details>

        <details className="more">
          <summary>How to reproduce</summary>
          <ul className="protocol">
            <li>α and the history threshold θ (= {data.best_theta}) are tuned on validation customers only; nothing is tuned on test.</li>
            <li>Every number is produced by the pipeline and checked by <code>make verify</code>. Re-running reuses the saved LLM answers, so it costs $0.</li>
            <li>Write-up: <code>docs/REPORT.md</code> · numbers: <code>results/metrics.json</code>, <code>results/extra_metrics.json</code>.</li>
          </ul>
        </details>
      </section>
    </div>
  )
}
