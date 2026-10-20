import React, { useEffect, useMemo, useState } from "react"
import { api } from "./api.js"
import ItemCard, { ProductImage, SkeletonCard, Stars } from "./components/ItemCard.jsx"
import ProductModal from "./components/ProductModal.jsx"
import AgentTrace from "./components/AgentTrace.jsx"
import ResultsView from "./components/ResultsView.jsx"
import AccountMenu, { Avatar, shortId } from "./components/AccountMenu.jsx"

const PAGE_SIZE = 12
const AGENTS = ["Reading your history", "Checking each game against your taste", "Summarising the best matches", "Ranking + GRU fusion"]
const EXAMPLES = ["open world rpg", "racing game for kids", "nintendo switch controller", "gaming headset under 50"]

const fmtDate = (ts) => (ts ? new Date(ts).toLocaleDateString(undefined, { year: "numeric", month: "short" }) : "")

function parseList(s) {
  const t = (s || "").trim().replace(/^\[|\]$/g, "").trim()
  if (!t) return []
  return t.split(/,\s*/).map((x) => x.replace(/^['"]|['"]$/g, "").trim()).filter(Boolean)
}

function tasteFromTrace(trace) {
  const lines = (trace?.user_understanding || "").split("\n")
  const get = (k) => (lines.find((l) => l.startsWith(k + ":")) || "").slice(k.length + 1).trim()
  return { intent: get("current_intent").replace(/\.+$/, ""), interests: parseList(get("long_term_interests")).slice(0, 6) }
}

/* ---------------- loading state while the agents run ---------------- */
function AgentsWorking() {
  const [step, setStep] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setStep((s) => (s + 1) % AGENTS.length), 2200)
    return () => clearInterval(t)
  }, [])
  return (
    <div className="working">
      <span className="spinner" />
      <div>
        <b>4 AI agents are building your recommendations</b>
        <div className="working-steps">
          {AGENTS.map((a, i) => (
            <span key={a} className={i === step ? "active" : ""}>{a}</span>
          ))}
        </div>
        <div className="muted small">Live LLM calls, usually 10–30 seconds.</div>
      </div>
    </div>
  )
}

/* ---------------- account picker (landing page) ---------------- */
function ChooseShopper({ users, onPick }) {
  const [showAll, setShowAll] = useState(false)
  const shown = showAll ? users : users.slice(0, 12)
  return (
    <section className="landing">
      <h1>Who's shopping today?</h1>
      <p className="muted">
        Each account is a real Amazon customer from the test set. Their history is their real reviews; the
        recommendations are produced live by the ARAG agents fused with our trained GRU.
      </p>
      <div className="shoppers">
        {shown.map((u) => (
          <button key={u.user_id} className="shopper" onClick={() => onPick(u.user_id)}>
            <Avatar id={u.user_id} size={52} />
            <span className="shopper-id">{shortId(u.user_id)}</span>
            <span className="muted small">{u.n_history} games reviewed</span>
            {u.top_platform && <span className="platform">{u.top_platform}</span>}
          </button>
        ))}
      </div>
      {!showAll && users.length > 12 && (
        <button className="link" onClick={() => setShowAll(true)}>Show all {users.length} customers</button>
      )}
    </section>
  )
}

/* ---------------- past purchases row ---------------- */
function HistoryRow({ history, onOpen }) {
  const games = history.filter((h) => h.source === "dataset")
  if (!games.length) return null
  return (
    <section className="section">
      <div className="section-head">
        <h2>Your past games</h2>
        <span className="muted small">{games.length} reviewed, newest first</span>
      </div>
      <div className="hscroll">
        {games.map((h) => (
          <button key={h.event_id} className="mini" onClick={() => onOpen(h)}>
            <ProductImage src={h.image} alt={h.title} size="sm" />
            <span className="mini-title">{h.title || h.item_id}</span>
            <span className="mini-meta">
              <span className="you-rated">You rated {"★".repeat(Math.round(h.rating || 0))}</span>
              <span className="muted">{fmtDate(h.ts)}</span>
            </span>
          </button>
        ))}
      </div>
    </section>
  )
}

/* ---------------- product grid with feedback ---------------- */
function useFeedback(user, session) {
  const [marks, setMarks] = useState({})
  const [toast, setToast] = useState("")
  useEffect(() => setMarks({}), [user])
  const act = (item_id, type, onDismiss) =>
    api.feedback(user, session, item_id, type).then(() => {
      setMarks((m) => ({ ...m, [item_id]: { ...(m[item_id] || {}), [type]: true } }))
      if (type === "dismiss") onDismiss?.(item_id)
      setToast(
        type === "dismiss"
          ? "Got it, we won't show that again."
          : type === "like"
          ? "Liked. Refresh to let the agents use it."
          : "Saved for later."
      )
      setTimeout(() => setToast(""), 2500)
    }).catch((e) => setToast(e.message))
  return { marks, toast, act }
}

function Grid({ items, startRank, fb, onOpen, onDismiss }) {
  return (
    <div className="grid">
      {items.map((it, i) => (
        <ItemCard
          key={it.item_id}
          item={it}
          rank={startRank != null ? startRank + i : null}
          onOpen={onOpen}
          liked={fb.marks[it.item_id]?.like}
          saved={fb.marks[it.item_id]?.save}
          onAction={(id, type) => fb.act(id, type, onDismiss)}
        />
      ))}
    </div>
  )
}

/* ---------------- home: recommended for you ---------------- */
function Home({ user, session, history, onOpen, onRun, run }) {
  const [items, setItems] = useState([])
  const [page, setPage] = useState(0)
  const [busy, setBusy] = useState(false)
  const [more, setMore] = useState(true)
  const [err, setErr] = useState("")
  const [showTrace, setShowTrace] = useState(false)
  const fb = useFeedback(user, session)

  const load = (p, append) => {
    setBusy(true); setErr("")
    api.feed(user, session, p, PAGE_SIZE)
      .then((r) => {
        setItems((prev) => (append ? [...prev, ...r.items] : r.items))
        setMore(r.items.length === PAGE_SIZE)
        setPage(p)
        if (r.run_id && !append) api.run(r.run_id).then(onRun).catch(() => {})
      })
      .catch((e) => setErr(e.message))
      .finally(() => setBusy(false))
  }
  useEffect(() => { if (session) load(0, false) }, [session])

  const taste = tasteFromTrace(run?.trace)
  const games = history.filter((h) => h.source === "dataset").length

  return (
    <>
      <section className="hero">
        <div>
          <div className="muted small">Welcome back, customer {shortId(user)}</div>
          <h1>Picked for you</h1>
          <p className="muted">
            Based on the {games} games you've reviewed.
            {taste.intent && <> What you seem to be after right now: <b>{taste.intent}</b>.</>}
          </p>
          {taste.interests.length > 0 && (
            <div className="chips">{taste.interests.map((x) => <span key={x} className="chip">{x}</span>)}</div>
          )}
        </div>
        <div className="hero-actions">
          <button onClick={() => load(0, false)} disabled={busy}>↻ Refresh</button>
          {run?.trace && (
            <button className="ghost" onClick={() => setShowTrace((s) => !s)}>
              {showTrace ? "Hide" : "How we picked these"}
            </button>
          )}
        </div>
      </section>

      {fb.toast && <div className="toast">{fb.toast}</div>}
      {err && <div className="err">{err}</div>}
      {showTrace && run?.trace && <AgentTrace run={run} />}

      <section className="section">
        {busy && items.length === 0 ? (
          <>
            <AgentsWorking />
            <div className="grid">{Array.from({ length: 8 }, (_, i) => <SkeletonCard key={i} />)}</div>
          </>
        ) : (
          <>
            <Grid items={items} startRank={1} fb={fb} onOpen={onOpen}
                  onDismiss={(id) => setItems((xs) => xs.filter((x) => x.item_id !== id))} />
            {!busy && items.length === 0 && !err && <p className="muted">No recommendations left for this customer.</p>}
            {more && items.length > 0 && (
              <div className="center">
                <button className="ghost" disabled={busy} onClick={() => load(page + 1, true)}>
                  {busy ? "Loading…" : "Show more"}
                </button>
              </div>
            )}
          </>
        )}
      </section>

      <HistoryRow history={history} onOpen={onOpen} />
    </>
  )
}

/* ---------------- search results ---------------- */
function SearchResults({ user, session, query, onOpen, onRun }) {
  const [items, setItems] = useState([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState("")
  const fb = useFeedback(user, session)

  useEffect(() => {
    if (!query) return
    setBusy(true); setErr(""); setItems([])
    api.search(user, session, query, PAGE_SIZE)
      .then((r) => {
        setItems(r.items)
        if (r.run_id) api.run(r.run_id).then(onRun).catch(() => {})
      })
      .catch((e) => setErr(e.message))
      .finally(() => setBusy(false))
  }, [query, session])

  return (
    <section className="section">
      <div className="section-head">
        <h2>Results for “{query}”</h2>
        <span className="muted small">re-ranked for this customer by ARAG + GRU</span>
      </div>
      {fb.toast && <div className="toast">{fb.toast}</div>}
      {err && <div className="err">{err}</div>}
      {busy ? (
        <>
          <AgentsWorking />
          <div className="grid">{Array.from({ length: 4 }, (_, i) => <SkeletonCard key={i} />)}</div>
        </>
      ) : (
        <Grid items={items} startRank={1} fb={fb} onOpen={onOpen}
              onDismiss={(id) => setItems((xs) => xs.filter((x) => x.item_id !== id))} />
      )}
      {!busy && !err && items.length === 0 && <p className="muted">No matching games.</p>}
    </section>
  )
}

/* ---------------- taste profile ---------------- */
function Profile({ user }) {
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState("")

  useEffect(() => { api.profile(user).then(setProfile).catch(() => setProfile(null)) }, [user])
  const generate = () => {
    setLoading(true); setMsg("")
    api.generateProfile(user).then(setProfile).catch((e) => setMsg(e.message)).finally(() => setLoading(false))
  }
  const remove = () => api.deleteProfile(user).then(() => { setProfile(null); setMsg("Profile deleted.") })
  const p = profile?.profile || null

  return (
    <section className="section narrow">
      <h1>Your taste profile</h1>
      <p className="muted">
        What the User Understanding agent learned from this customer's history. It's stored with its source and
        can be deleted at any time.
      </p>
      <div className="row">
        <button onClick={generate} disabled={loading}>{loading ? "Analysing history…" : p ? "Update profile" : "Build my profile"}</button>
        <button className="ghost" onClick={remove} disabled={!profile}>Delete profile</button>
      </div>
      {msg && <div className="toast">{msg}</div>}
      {p ? (
        <div className="profile">
          {p.current_intent && <div className="pf-intent"><span>Looking for right now</span><b>{p.current_intent}</b></div>}
          {p.summary && <p>{p.summary}</p>}
          {(p.long_term_interests || []).length > 0 && (
            <>
              <div className="pf-label">Long-term interests</div>
              <div className="chips">{p.long_term_interests.map((x, i) => <span key={i} className="chip">{x}</span>)}</div>
            </>
          )}
          {(p.avoid || []).length > 0 && (
            <>
              <div className="pf-label">Tends to avoid</div>
              <div className="chips">{p.avoid.map((x, i) => <span key={i} className="chip chip-avoid">{x}</span>)}</div>
            </>
          )}
          <div className="muted small">
            version {profile.version} · {profile.provenance?.model} · built from {profile.provenance?.events_used} events · {profile.updated_at}
          </div>
        </div>
      ) : (
        !loading && <p className="muted">No profile yet.</p>
      )}
    </section>
  )
}

/* ---------------- research page ---------------- */
function Research({ stats, nDemo }) {
  const [results, setResults] = useState(null)
  useEffect(() => { api.results().then(setResults).catch(() => {}) }, [])
  return (
    <section className="section research">
      <ResultsView data={results} stats={stats} nDemo={nDemo} />
    </section>
  )
}

/* ---------------- app shell ---------------- */
export default function App() {
  const [users, setUsers] = useState([])
  const [stats, setStats] = useState(null)
  const [user, setUser] = useState("")
  const [session, setSession] = useState("")
  const [history, setHistory] = useState([])
  const [view, setView] = useState(() => (window.location.hash === "#research" ? "research" : "home"))
  const [query, setQuery] = useState("")
  const [draft, setDraft] = useState("")
  const [run, setRun] = useState(null)
  const [open, setOpen] = useState(null)
  const [err, setErr] = useState("")

  useEffect(() => { api.users().then(setUsers).catch((e) => setErr(e.message)) }, [])
  useEffect(() => { api.dataset().then(setStats).catch(() => {}) }, [])

  const pick = (uid) => {
    setUser(uid); setRun(null); setErr(""); setSession(""); setHistory([]); setView("home")
    if (!uid) return
    Promise.all([
      api.createSession(uid).then((r) => setSession(r.session_id)),
      api.history(uid).then((r) => setHistory(r.items)),
    ]).catch((e) => setErr(e.message))
  }
  const search = (q) => {
    const t = q.trim()
    if (!t || !user) return
    setQuery(t); setView("search")
  }
  const me = useMemo(() => users.find((u) => u.user_id === user), [users, user])

  return (
    <div className="shell">
      <header className="nav">
        <div className="nav-inner">
          <button className="logo" onClick={() => (user ? setView("home") : null)}>
            <span className="logo-mark">▶</span> GameShelf
          </button>
          <form className="nav-search" onSubmit={(e) => { e.preventDefault(); search(draft) }}>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={user ? "Search games, consoles, accessories…" : "Choose a customer to start"}
              disabled={!user}
            />
            <button disabled={!user}>Search</button>
          </form>
          <div className="nav-links">
            {user && <button className={view === "home" ? "on" : ""} onClick={() => setView("home")}>For you</button>}
            {user && <button className={view === "profile" ? "on" : ""} onClick={() => setView("profile")}>Taste profile</button>}
            <button className={view === "research" ? "on" : ""} onClick={() => setView("research")}>Research</button>
            <AccountMenu users={users} current={user} onPick={pick} />
          </div>
        </div>
        {user && view === "search" && (
          <div className="nav-chips">
            {EXAMPLES.map((ex) => <button key={ex} className="chip chip-btn" onClick={() => { setDraft(ex); search(ex) }}>{ex}</button>)}
          </div>
        )}
      </header>

      <main className="main">
        {err && <div className="err">{err}</div>}
        {view === "research" ? (
          <Research stats={stats} nDemo={users.length} />
        ) : !user ? (
          <ChooseShopper users={users} onPick={pick} />
        ) : view === "profile" ? (
          <Profile user={user} />
        ) : view === "search" ? (
          <SearchResults key={user} user={user} session={session} query={query} onOpen={setOpen} onRun={setRun} />
        ) : (
          <Home key={user} user={user} session={session} history={history} onOpen={setOpen} onRun={setRun} run={run} />
        )}
      </main>

      {open && <ProductModal item={open} trace={run?.trace} onClose={() => setOpen(null)} />}

      <footer className="footer">
        <span>GameShelf is a course-project demo of ARAG + GRU on Amazon Reviews 2023 (Video Games).</span>
        {me && <span>Signed in as {shortId(me.user_id)} · {me.n_history} reviews</span>}
        {run && <span>run {run.run_id.slice(0, 8)} · {run.trace?.alpha != null && `α=${run.trace.alpha}`}</span>}
      </footer>
    </div>
  )
}
