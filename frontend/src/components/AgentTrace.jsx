function parseList(s) {
  const t = (s || "").trim().replace(/^\[|\]$/g, "").trim()
  if (!t) return []
  return t
    .split(/,\s*/)
    .map((x) => x.replace(/^['"]|['"]$/g, "").trim())
    .filter(Boolean)
}

export default function AgentTrace({ run }) {
  if (!run || !run.trace) return null
  const t = run.trace
  const uua = (t.user_understanding || "").split("\n")
  const get = (key) => {
    const line = uua.find((l) => l.startsWith(key + ":"))
    return line ? line.slice(key.length + 1).trim() : ""
  }
  const interests = parseList(get("long_term_interests"))
  const avoid = parseList(get("avoid"))
  const intent = get("current_intent")
  const summary = get("summary")

  const accepted = (t.nli || []).filter((n) => n.accepted)
  const anyFallback = t.fallbacks && Object.values(t.fallbacks).some(Boolean)
  const anyRepair = t.repairs && t.repairs.unknown + t.repairs.duplicate + t.repairs.missing > 0

  return (
    <section className="trace">
      <div className="trace-head">
        <h2>How this ranking was built</h2>
        <div className="badges">
          <span className="badge">ARAG · θ={t.theta}</span>
          <span className="badge">GRU fusion · α={t.alpha}</span>
        </div>
      </div>

      <ol className="steps">
        <li>
          <div className="step-title">1 · User understanding agent</div>
          {intent && (
            <p className="step-lead">
              Current intent: <b>{intent}</b>
            </p>
          )}
          {summary && <p>{summary}</p>}
          {interests.length > 0 && (
            <div className="chips">
              {interests.map((x, i) => (
                <span key={i} className="chip">
                  {x}
                </span>
              ))}
            </div>
          )}
          {avoid.length > 0 && (
            <div className="chips">
              {avoid.map((x, i) => (
                <span key={i} className="chip chip-avoid">
                  avoids: {x}
                </span>
              ))}
            </div>
          )}
        </li>

        <li>
          <div className="step-title">2 · NLI relevance check</div>
          <p className="step-lead">
            {accepted.length} of {(t.nli || []).length} candidates judged relevant (score ≥ {t.theta}).
          </p>
          <ul className="nli-list">
            {(t.nli || []).map((n, i) => (
              <li key={i} className={n.accepted ? "nli-accepted" : "nli-rejected"}>
                <span className={`label label-${n.label}`}>{n.label}</span>
                <span className="nli-score">{n.score}</span>
                <span className="nli-title">{n.title}</span>
                <span className="nli-why">{n.rationale}</span>
              </li>
            ))}
          </ul>
          {t.nli_missing > 0 && <span className="warn">missing after re-ask: {t.nli_missing}</span>}
        </li>

        <li>
          <div className="step-title">3 · Context summary</div>
          {t.csa_empty_fallback && (
            <span className="warn">no candidate passed θ — used top-3 fallback</span>
          )}
          <p>{t.context_summary || "(none)"}</p>
        </li>

        <li>
          <div className="step-title">4 · Item ranker agent → fused with GRU</div>
          <ol className="final-ranking">
            {(t.final_ranking || []).slice(0, 12).map((it, i) => (
              <li key={i}>
                <span className="fr-rank">{i + 1}</span> {it.title}
              </li>
            ))}
          </ol>
          {(t.final_ranking || []).length > 12 && (
            <p className="muted">+{(t.final_ranking || []).length - 12} more candidates…</p>
          )}
          {anyFallback || anyRepair ? (
            <div className="badges">
              {anyFallback && <span className="badge warn">fallback used</span>}
              {anyRepair && <span className="badge warn">output repaired</span>}
            </div>
          ) : (
            <span className="ok">no fallbacks or repairs needed</span>
          )}
        </li>
      </ol>
    </section>
  )
}
