import { useEffect } from "react"
import { ProductImage, Stars } from "./ItemCard.jsx"

export default function ProductModal({ item, trace, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])
  if (!item) return null
  const nli = (trace?.nli || []).find((n) => n.item_id === item.item_id)

  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label={item.title}>
        <button className="modal-x" onClick={onClose} aria-label="Close">✕</button>
        <div className="modal-grid">
          <ProductImage src={item.image} alt={item.title} size="lg" />
          <div className="modal-info">
            {item.platform && <span className="platform">{item.platform}</span>}
            <h2>{item.title}</h2>
            {item.store && <div className="muted small">by {item.store}</div>}
            <Stars value={item.avg_rating} count={item.n_ratings} />
            {item.price != null && <div className="price big">${item.price.toFixed(2)}</div>}

            {(item.reason || nli) && (
              <div className="why-box">
                <div className="why-head">Why we recommended this</div>
                {item.reason && <p>{item.reason}</p>}
                {nli && (
                  <p className="small">
                    Relevance agent: <span className={`label label-${nli.label}`}>{nli.label}</span> score{" "}
                    <b>{nli.score}</b>/10, {nli.rationale}
                  </p>
                )}
              </div>
            )}

            {item.features?.length > 0 && (
              <ul className="features">
                {item.features.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            )}
            {item.about && <p className="about">{item.about}</p>}
            <div className="muted small">Product ID {item.item_id}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
