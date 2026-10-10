export default function ItemCard({ item, rank, onAction, busy }) {
  return (
    <article className="card">
      <div className="card-top">
        {rank != null && <span className="rank">#{rank}</span>}
        <h3 className="card-title">{item.title}</h3>
      </div>
      <div className="meta">
        {item.category && <span className="tag">{item.category}</span>}
        {item.price != null && <span className="price">${item.price.toFixed(2)}</span>}
        {item.score != null && <span className="score">match {(item.score * 100).toFixed(0)}%</span>}
      </div>
      {item.reason && (
        <p className="reason">
          <span className="reason-label">Why this?</span> {item.reason}
        </p>
      )}
      {item.evidence && <blockquote className="evidence">{item.evidence}</blockquote>}
      <div className="actions">
        <button className="btn-like" onClick={() => onAction(item.item_id, "like")} disabled={busy}>
          Like
        </button>
        <button className="btn-dismiss" onClick={() => onAction(item.item_id, "dismiss")} disabled={busy}>
          Not interested
        </button>
        <button className="btn-save" onClick={() => onAction(item.item_id, "save")} disabled={busy}>
          Save
        </button>
      </div>
    </article>
  )
}
