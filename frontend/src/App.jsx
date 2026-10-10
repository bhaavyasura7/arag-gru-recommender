import React, { useEffect, useState } from "react"
import { api } from "./api.js"
import ItemCard from "./components/ItemCard.jsx"
import AgentTrace from "./components/AgentTrace.jsx"
import ResultsView from "./components/ResultsView.jsx"

const PIPELINE = ["User understanding", "NLI relevance", "Context summary", "Item ranking", "GRU fusion"]

function fmtDate(ts) {
  if (!ts) return ""
  const d = new Date(ts)
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short" })
}

function UserContext({ history }) {
  const [open, setOpen] = useState(true)
  if (!history || history.length === 0) return null
  const recent = history.filter((h) => h.source === "dataset").slice(0, 8)
  return (
    <div className="context">
      <button className="context-head" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} This user's recent history ({recent.length} shown)
      </button>
      {open && (
        <ul className="history">
          {recent.map((h, i) => (
            <li key={i}>
              <span className="h-title">{(h.title || h.item_id).slice(0, 70)}</span>
              <span className="h-meta">
                {h.main_category} · ★{Number(h.rating).toFixed(0)} · {fmtDate(h.ts)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function FeedTab({ user, session, onRun }) {
  const [items, setItems] = useState([])
  const [page, setPage] = useState(0)
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState("")

  const load = (p) => {
    setBusy(true)
    api.feed(user, session, p)
      .then((r) => {
        setItems(r.items)
        if (r.run_id) api.run(r.run_id).then(onRun).catch(() => {})
      })
      .catch((e) => setToast(e.message))
      .finally(() => setBusy(false))
  }
  useEffect(() => { if (session) load(0) }, [session])

  const act = (item_id, type) => {
    api.feedback(user, session, item_id, type).then(() => {
      setToast(type === "dismiss" ? "Removed — the next feed won't show it." : `Feedback recorded: ${type}.`)
      load(page)
    })
  }

  return (
    <div>
      <p className="lead">
        Personalised picks ranked by the <b>ARAG</b> agent pipeline fused with the <b>GRU</b> sequential model.
      </p>
      {toast && <div className="toast" onClick={() => setToast("")}>{toast}</div>}
      <div className="cards">
        {items.map((it, i) => (
          <ItemCard key={it.item_id} item={it} rank={page * 10 + i + 1} onAction={act} busy={busy} />
        ))}
      </div>
      {items.length === 0 && !busy && <p className="muted">No more items for this user.</p>}
      <div className="pager">
        <button disabled={page === 0 || busy} onClick={() => { const p = page - 1; setPage(p); load(p) }}>
          ← Previous
        </button>
        <span>page {page + 1}</span>
        <button disabled={busy || items.length === 0} onClick={() => { const p = page + 1; setPage(p); load(p) }}>
          Next →
        </button>
      </div>
    </div>
  )
}

const EXAMPLES = ["mechanical gaming keyboard rgb", "nintendo switch controller", "gaming headset under 50"]

function SearchTab({ user, session, onRun }) {
  const [query, setQuery] = useState("")
  const [items, setItems] = useState([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState("")

  const run = () => {
    if (!query.trim()) return
    setBusy(true); setErr("")
    api.search(user, session, query)
      .then((r) => {
        setItems(r.items)
        if (r.run_id) api.run(r.run_id).then(onRun).catch(() => {})
      })
      .catch((e) => setErr(e.message))
      .finally(() => setBusy(false))
  }

  return (
    <div>
      <p className="lead">Search the catalogue; results are re-ranked for this user by ARAG + GRU.</p>
      <div className="searchbar">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="e.g. wireless gaming mouse"
        />
        <button onClick={run} disabled={busy}>{busy ? "Ranking…" : "Search"}</button>
      </div>
      <div className="chips">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip chip-btn" onClick={() => setQuery(ex)}>{ex}</button>
        ))}
      </div>
      {err && <div className="err">{err}</div>}
      <div className="cards">
        {items.map((it, i) => <ItemCard key={it.item_id} item={it} rank={i + 1} onAction={() => {}} />)}
      </div>
    </div>
  )
}

function ProfileTab({ user }) {
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState("")

  const load = () =>
    api.profile(user).then(setProfile).catch(() => setProfile(null))
  useEffect(() => { load() }, [user])

  const generate = () => {
    setLoading(true); setMsg("")
    api.generateProfile(user).then(setProfile).catch((e) => setMsg(e.message)).finally(() => setLoading(false))
  }
  const remove = () => api.deleteProfile(user).then(() => { setProfile(null); setMsg("Profile deleted.") })

  const p = profile?.profile || null
  return (
    <div>
      <p className="lead">
        What the <b>User Understanding Agent</b> inferred from this user's history. Stored in the database with
        provenance and deletable.
      </p>
      <div className="row">
        <button onClick={generate} disabled={loading}>{loading ? "Analysing history…" : "Generate / update profile"}</button>
        <button className="ghost" onClick={remove} disabled={!profile}>Delete profile</button>
      </div>
      {msg && <div className="toast" onClick={() => setMsg("")}>{msg}</div>}
      {p ? (
        <div className="profile">
          {p.current_intent && <div className="pf-intent"><span>Current intent</span><b>{p.current_intent}</b></div>}
          {p.summary && <p className="pf-summary">{p.summary}</p>}
          {(p.long_term_interests || []).length > 0 && (
            <>
              <div className="pf-label">Long-term interests</div>
              <div className="chips">{p.long_term_interests.map((x, i) => <span key={i} className="chip">{x}</span>)}</div>
            </>
          )}
          {(p.avoid || []).length > 0 && (
            <>
              <div className="pf-label">Avoids</div>
              <div className="chips">{p.avoid.map((x, i) => <span key={i} className="chip chip-avoid">{x}</span>)}</div>
            </>
          )}
          <div className="pf-prov">
            version {profile.version} · model {profile.provenance?.model} · {profile.provenance?.events_used} events used ·{" "}
            {profile.updated_at}
          </div>
        </div>
      ) : (
        <p className="muted">No profile stored yet for this user.</p>
      )}
    </div>
  )
}

export default function App() {
  const [users, setUsers] = useState([])
  const [user, setUser] = useState("")
  const [session, setSession] = useState("")
  const [history, setHistory] = useState([])
  const [tab, setTab] = useState("feed")
  const [run, setRun] = useState(null)
  const [results, setResults] = useState(null)
  const [err, setErr] = useState("")

  useEffect(() => { api.users().then(setUsers).catch((e) => setErr(e.message)) }, [])
  useEffect(() => { if (tab === "benchmarks" && !results) api.results().then(setResults).catch(() => {}) }, [tab])

  const pick = (uid) => {
    setUser(uid); setRun(null); setErr("")
    Promise.all([
      api.createSession(uid).then((r) => setSession(r.session_id)),
      api.history(uid).then((r) => setHistory(r.items)),
    ]).catch((e) => setErr(e.message))
  }

  return (
    <div className="app">
      <header className="header">
        <h1>ARAG + GRU Recommender</h1>
        <p className="sub">
          An agentic recommender on Amazon Video Games: an LLM agent pipeline ranks candidates, and a trained
          GRU sequential model is fused into the final ranking.
        </p>
        <div className="pipeline">
          {PIPELINE.map((s, i) => (
            <React.Fragment key={s}>
              <span className="pipe-step">{s}</span>
              {i < PIPELINE.length - 1 && <span className="pipe-arrow">→</span>}
            </React.Fragment>
          ))}
        </div>
      </header>

      <div className="userbar">
        <label htmlFor="user">Demo user</label>
        <select id="user" value={user} onChange={(e) => pick(e.target.value)}>
          <option value="">select a user…</option>
          {users.map((u) => (
            <option key={u.user_id} value={u.user_id}>
              {u.user_id.length > 22 ? u.user_id.slice(0, 20) + "…" : u.user_id}
            </option>
          ))}
        </select>
        {session && <span className="session-ok">session active</span>}
      </div>
      {err && <div className="err">{err}</div>}

      {user && <UserContext history={history} />}

      <nav className="tabs">
        {[["feed", "Feed"], ["search", "Search"], ["profile", "Profile"], ["benchmarks", "Benchmarks"]].map(([k, label]) => (
          <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{label}</button>
        ))}
      </nav>

      <main className="content">
        {!user && tab !== "benchmarks" && <p className="muted">Select a demo user above to begin.</p>}
        {user && tab === "feed" && <FeedTab user={user} session={session} onRun={setRun} />}
        {user && tab === "search" && <SearchTab user={user} session={session} onRun={setRun} />}
        {user && tab === "profile" && <ProfileTab user={user} />}
        {tab === "benchmarks" && <ResultsView data={results} />}
      </main>

      {run && run.trace && <AgentTrace run={run} />}

      <footer className="footer">
        {run && <span>run {run.run_id.slice(0, 8)} · {run.kind}</span>}
        <span>No fabricated data — every number is generated by the pipeline and audited with `make verify`.</span>
      </footer>
    </div>
  )
}
