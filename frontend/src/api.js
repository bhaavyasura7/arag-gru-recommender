const BASE = ""

async function jsonFetch(url, options = {}) {
  const res = await fetch(BASE + url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  health: () => jsonFetch("/health"),
  users: () => jsonFetch("/users"),
  createSession: (user_id) =>
    jsonFetch("/sessions", { method: "POST", body: JSON.stringify({ user_id }) }),
  feed: (user_id, session_id, page = 0, pageSize = 10) =>
    jsonFetch(`/feed?user_id=${user_id}&session_id=${session_id}&page=${page}&page_size=${pageSize}`),
  search: (user_id, session_id, query, limit = 5) =>
    jsonFetch("/search", {
      method: "POST",
      body: JSON.stringify({ user_id, session_id, query, limit }),
    }),
  feedback: (user_id, session_id, item_id, event_type) =>
    jsonFetch("/feedback", {
      method: "POST",
      body: JSON.stringify({ user_id, session_id, item_id, event_type }),
    }),
  profile: (user_id) => jsonFetch(`/users/${user_id}/profile`),
  deleteProfile: (user_id) =>
    jsonFetch(`/users/${user_id}/profile`, { method: "DELETE" }),
  run: (run_id) => jsonFetch(`/runs/${run_id}`),
  replay: (run_id) =>
    jsonFetch(`/runs/${run_id}/replay`, { method: "POST", body: "{}" }),
  results: () => jsonFetch("/results"),
}
