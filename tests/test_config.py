from pathlib import Path

from arag.config import REPO_ROOT, load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.category == "Video_Games"
    assert cfg.llm_provider in ("ollama", "openai")
    assert cfg.llm_model
    assert cfg.seed == 42


def test_repo_root_contains_config():
    assert (REPO_ROOT / "configs" / "config.yaml").exists()


def test_paths_resolve_relative_to_repo_root():
    cfg = load_config()
    p = cfg.path("artifacts")
    assert p.is_absolute()
    assert p.parent == REPO_ROOT


def test_theta_and_alpha_grids():
    cfg = load_config()
    assert cfg.get("agents.theta_grid") == [4, 5, 6, 7]
    alpha = cfg.get("fusion.alpha_grid")
    assert alpha[0] == 0.0
    assert alpha[-1] == 1.0
