import { useEffect, useRef, useState } from "react"

const HUES = [243, 199, 160, 28, 330, 270, 12, 186]

export function Avatar({ id, size = 36 }) {
  let h = 0
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) >>> 0
  const hue = HUES[h % HUES.length]
  return (
    <span
      className="avatar"
      style={{
        width: size, height: size, fontSize: size * 0.36,
        background: `hsl(${hue} 70% 92%)`, color: `hsl(${hue} 60% 32%)`,
      }}
    >
      {id.slice(2, 4)}
    </span>
  )
}

export const shortId = (id) => (id.length > 10 ? id.slice(0, 8) + "…" : id)

const SORTS = [
  ["most", "Most games"],
  ["fewest", "Fewest games"],
]

export default function AccountMenu({ users, current, onPick }) {
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState("")
  const [sort, setSort] = useState("most")
  const [phoneTop, setPhoneTop] = useState(null)
  const ref = useRef(null)
  const me = users.find((u) => u.user_id === current)

  useEffect(() => {
    if (!open) return
    const onDown = (e) => ref.current && !ref.current.contains(e.target) && setOpen(false)
    const onKey = (e) => e.key === "Escape" && setOpen(false)
    document.addEventListener("mousedown", onDown)
    document.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("mousedown", onDown)
      document.removeEventListener("keydown", onKey)
    }
  }, [open])

  const needle = q.trim().toLowerCase()
  const list = users
    .filter((u) => !needle || u.user_id.toLowerCase().includes(needle) || (u.top_platform || "").toLowerCase().includes(needle))
    .sort((a, b) => (sort === "most" ? b.n_history - a.n_history : a.n_history - b.n_history))

  const toggle = () => {
    // on phones the panel spans the screen just below the button instead of hanging off it
    const r = ref.current?.getBoundingClientRect()
    setPhoneTop(window.innerWidth < 720 && r ? r.bottom + 8 : null)
    setOpen((o) => !o)
  }

  const choose = (id) => {
    setOpen(false)
    setQ("")
    onPick(id)
  }

  return (
    <div className="acct" ref={ref}>
      <button className={`acct-btn ${open ? "open" : ""}`} onClick={toggle} aria-expanded={open}>
        {me ? (
          <>
            <Avatar id={me.user_id} size={30} />
            <span className="acct-text">
              <span className="acct-hello">Hello, customer</span>
              <span className="acct-name">{shortId(me.user_id)}</span>
            </span>
          </>
        ) : (
          <span className="acct-text">
            <span className="acct-hello">Hello, sign in</span>
            <span className="acct-name">Choose account</span>
          </span>
        )}
        <span className="acct-caret">▾</span>
      </button>

      {open && (
        <div
          className="acct-panel"
          role="menu"
          style={phoneTop != null ? { position: "fixed", top: phoneTop, left: 16, right: 16, width: "auto" } : undefined}
        >
          {me && (
            <div className="acct-me">
              <Avatar id={me.user_id} size={44} />
              <div className="acct-me-info">
                <div className="acct-me-id">{me.user_id}</div>
                <div className="muted small">
                  {me.n_history} games reviewed{me.top_platform && <> · mostly {me.top_platform}</>}
                </div>
              </div>
              <button className="acct-out" onClick={() => choose("")}>Sign out</button>
            </div>
          )}
          <div className="acct-tools">
            <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by customer ID or platform…" />
            <div className="acct-sort">
              {SORTS.map(([k, label]) => (
                <button key={k} className={sort === k ? "on" : ""} onClick={() => setSort(k)}>{label}</button>
              ))}
            </div>
          </div>
          <div className="acct-list">
            {list.map((u) => (
              <button
                key={u.user_id}
                className={`acct-row ${u.user_id === current ? "current" : ""}`}
                onClick={() => choose(u.user_id)}
              >
                <Avatar id={u.user_id} size={32} />
                <span className="acct-row-main">
                  <span className="acct-row-id">{shortId(u.user_id)}</span>
                  <span className="muted small">{u.top_platform || "Mixed platforms"}</span>
                </span>
                <span className="acct-count">{u.n_history}<span className="muted small"> games</span></span>
                {u.user_id === current && <span className="acct-check">✓</span>}
              </button>
            ))}
            {list.length === 0 && <div className="muted small acct-empty">No customer matches “{q}”.</div>}
          </div>
          <div className="acct-foot muted small">{users.length} real test customers · Amazon Reviews 2023</div>
        </div>
      )}
    </div>
  )
}
