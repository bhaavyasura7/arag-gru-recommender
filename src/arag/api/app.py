"""FastAPI app: wires routes + static frontend, seeds the DB on ``make app``."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from arag import util
from arag.api.routes import router
from arag.api.services import RecommenderService
from arag.config import get_config
from arag.llm.client import LLMError
from arag.memory import repo


def create_app(service: RecommenderService | None = None, db_path: str | None = None) -> FastAPI:
    cfg = get_config()
    db_path = db_path or str(cfg.path("app_db"))
    service = service or RecommenderService(cfg, db_path)

    app = FastAPI(title="ARAG + GRU Recommender")
    app.state.service = service
    app.include_router(router)

    @app.exception_handler(LLMError)
    async def llm_error_handler(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    dist = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="static")

    return app


def main() -> None:
    logger = util.setup_logging("arag.api.app")
    cfg = get_config()
    db_path = cfg.path("app_db")

    conn = repo.connect(db_path)
    has_users = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"] > 0
    conn.close()
    if not has_users:
        logger.info("seeding DB at %s ...", db_path)
        repo.seed_db(cfg, db_path)
    else:
        logger.info("DB already seeded (%s)", db_path)

    service = RecommenderService(cfg, db_path)
    app = create_app(service=service, db_path=str(db_path))

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
