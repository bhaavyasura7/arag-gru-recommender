export function Stars({ value, count }) {
  if (value == null) return null
  const full = Math.round(Number(value))
  return (
    <span className="stars" title={`${value} out of 5`}>
      <span className="stars-on">{"★".repeat(full)}</span>
      <span className="stars-off">{"★".repeat(5 - full)}</span>
      {count != null && <span className="stars-n">{Number(count).toLocaleString()}</span>}
    </span>
  )
}

export function ProductImage({ src, alt, size = "md" }) {
  return (
    <div className={`pimg pimg-${size}`}>
      {src ? <img src={src} alt={alt} loading="lazy" /> : <span className="pimg-none">No image</span>}
    </div>
  )
}

export default function ItemCard({ item, rank, onAction, onOpen, liked, saved }) {
  return (
    <article className="card" onClick={() => onOpen(item)}>
      {rank != null && <span className="rank">#{rank}</span>}
      <ProductImage src={item.image} alt={item.title} />
      <div className="card-body">
        {item.platform && <span className="platform">{item.platform}</span>}
        <h3 className="card-title">{item.title}</h3>
        <Stars value={item.avg_rating} count={item.n_ratings} />
        <div className="price-row">
          {item.price != null ? <span className="price">${item.price.toFixed(2)}</span> : <span className="muted small">Price unavailable</span>}
        </div>
        {item.reason && (
          <p className="reason">
            <b>Why: </b>
            {item.reason}
          </p>
        )}
      </div>
      {onAction && (
        <div className="actions" onClick={(e) => e.stopPropagation()}>
          <button className={liked ? "on" : ""} onClick={() => onAction(item.item_id, "like")} title="Like">
            {liked ? "♥ Liked" : "♡ Like"}
          </button>
          <button className={saved ? "on" : ""} onClick={() => onAction(item.item_id, "save")} title="Save for later">
            {saved ? "✓ Saved" : "Save"}
          </button>
          <button className="ghost" onClick={() => onAction(item.item_id, "dismiss")} title="Not interested">
            ✕
          </button>
        </div>
      )}
    </article>
  )
}

export function SkeletonCard() {
  return (
    <article className="card skeleton">
      <div className="pimg pimg-md" />
      <div className="card-body">
        <div className="sk-line" />
        <div className="sk-line short" />
        <div className="sk-line" />
      </div>
    </article>
  )
}
