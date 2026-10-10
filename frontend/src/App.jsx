import React, { useEffect, useState } from "react"
import { api } from "./api.js"

function Card({ item, onAction }) {
  return (
    <div className="card">
      <div className="card-head">
        <h3>{item.title}</h3>
        {item.price != null && <span className="price">${item.price.toFixed(2)}</span>}
      </div>
      <div className="cat">{item.category}</div>
      {item.reason && <div className="reason"><b>why:</b> {item.reason}</div>}
      {item.evidence && <div className="evidence">“{item.evidence}”</div>}
      <div className="actions">
        <button onClick={() => onAction(item.item_id, "like")}>Like</button>
        <button onClick={() => onAction(item.item_id, "dismiss")}>Dismiss</button>
        <button onClick={() => onAction(item.item_id, "save")}>Save</button>
      </div>
    </div>
  )
}

function FeedTab({ user, session, onTrace }) {
  const [items, setItems] = useState([])
  const [page, setPage] = useState(0)
  const [err, setErr] = useState("")

  const load = (p = 0) => {
    setErr("")
    api.feed(user, session, p).then((r) => {
      setItems(r.items)
      onTrace(r.run_id)
    }).catch((e) => setErr(e.message))
  }
  useEffect(() => { if (session) load(0) }, [session])

  const act = (item_id, type) => {
    api.feedback(user, session, item_id, type).then(() => load(page)).catch((e) => setErr(e.message))
  }

  return (
    <div>
      {err && <div className="err">{err}</div>}
      <div className="toolbar">
        <button disabled={page === 0} onClick={() => { setPage(page - 1); load(page - 1) }}>Prev</button>
        <span>page {page + 1}</span>
        <button onClick={() => { setPage(page + 1); load(page + 1) }}>Next</button>
      </div>
      <div className="grid">
        {items.map((it) => <Card key={it.item_id} item={it} onAction={act} />)}
      </div>
    </div>
  )
}

function SearchTab({ user, session, onTrace }) {
  const [query, setQuery] = useState("")
  const [items, setItems] = useState([])
  const [err, setErr] = useState("")

  const run = () => {
    setErr("")
    api.search(user, session, query).then((r) => {
      setItems(r.items)
      onTrace(r.run_id)
    }).catch((e) => setErr(e.message))
  }
  return (
    <div>
      <div className="toolbar">
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="search products..." />
        <button onClick={run}>Search</button>
      </div>
      {err && <div className="err">{err}</div>}
      <div className="grid">
        {items.map((it) => <Card key={it.item_id} item={it} onAction={() => {}} />)}
      </div>
    </div>
  )
}

function ProfileTab({ user }) {
  const [profile, setProfile] = useState(null)
  const [err, setErr] = useState("")
  const load = () => api.profile(user).then(setProfile).catch((e) => setErr(e.message))
  useEffect(load, [user])
  const del = () => api.deleteProfile(user).then(() => { setProfile(null); setErr("") }).catch((e) => setErr(e.message))
  return (
    <div>
      <button onClick={load}>Refresh</button>
      <button onClick={del}>Delete profile</button>
      {err && <div className="err">{err}</div>}
      {profile ? (
        <pre className="profile">{JSON.stringify(profile, null, 2)}</pre>
      ) : <p>No stored profile for this user.</p>}
    </div>
  )
}

function RunTrace({ runId }) {
  const [run, setRun] = useState(null)
  useEffect(() => {
    if (runId) api.run(runId).then(setRun).catch(() => setRun(null))
  }, [runId])
  if (!run) return <div className="drawer">no run selected</div>
  return (
    <div className="drawer">
      <h3>Run trace</h3>
      <div>kind: {run.kind} · user: {run.user_id}</div>
      <pre>{JSON.stringify(run.request_json, null, 2)}</pre>
      <h4>Ranked items</h4>
      <ol>{run.items.map((i) => <li key={i.item_id}>{i.item_id} (rank {i.rank})</li>)}</ol>
    </div>
  )
}

function ResultsTab() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState("")
  useEffect(() => { api.results().then(setData).catch((e) => setErr(e.message)) }, [])
  if (err) return <div className="err">{err}</div>
  if (!data) return <p>loading results...</p>
  const metrics = data.metrics || {}
  const systems = Object.keys(metrics).filter((k) => !k.includes("_vs_"))
  const names = ["hit@5", "ndcg@5", "hit@10", "ndcg@10", "mrr"]
  return (
    <div>
      <p>best θ = {data.best_theta}, best α = {data.best_alpha}</p>
      <table>
        <thead><tr><th>system</th>{names.map((n) => <th key={n}>{n}</th>)}</tr></thead>
        <tbody>
          {systems.map((s) => (
            <tr key={s}>
              <td>{s}</td>
              {names.map((n) => <td key={n}>{(metrics[s][n]?.mean ?? 0).toFixed(3)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <p>α curve (test): {JSON.stringify(data.alpha_curve_test)}</p>
    </div>
  )
}

export default function App() {
  const [users, setUsers] = useState([])
  const [user, setUser] = useState("")
  const [session, setSession] = useState("")
  const [tab, setTab] = useState("feed")
  const [runId, setRunId] = useState("")
  const [err, setErr] = useState("")

  useEffect(() => { api.users().then(setUsers).catch((e) => setErr(e.message)) }, [])

  const pick = (uid) => {
    setUser(uid)
    setErr("")
    api.createSession(uid).then((r) => setSession(r.session_id)).catch((e) => setErr(e.message))
  }

  return (
    <main>
      <h1>ARAG + GRU Recommender</h1>
      <div className="picker">
        <label>User</label>
        <select value={user} onChange={(e) => pick(e.target.value)}>
          <option value="">select...</option>
          {users.map((u) => <option key={u.user_id} value={u.user_id}>{u.user_id.slice(0, 12)}…</option>)}
        </select>
        {session && <span className="session">session {session.slice(0, 8)}…</span>}
      </div>
      {err && <div className="err">{err}</div>}

      <nav className="tabs">
        {["feed", "search", "profile", "results"].map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{t}</button>
        ))}
      </nav>

      {tab === "feed" && user && <FeedTab user={user} session={session} onTrace={setRunId} />}
      {tab === "search" && user && <SearchTab user={user} session={session} onTrace={setRunId} />}
      {tab === "profile" && user && <ProfileTab user={user} />}
      {tab === "results" && <ResultsTab />}

      <RunTrace runId={runId} />
    </main>
  )
}
