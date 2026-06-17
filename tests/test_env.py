"""arcstore._env.cache_dir: local-scratch-root precedence (torch-free)."""
from arcstore._env import cache_dir


def test_cache_dir_env_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCSTORE_CACHE_DIR", str(tmp_path / "root"))
    p = cache_dir("sub")
    assert p == tmp_path / "root" / "sub"
    assert p.is_dir()


def test_cache_dir_create_false(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCSTORE_CACHE_DIR", str(tmp_path / "root2"))
    p = cache_dir("sub", create=False)
    assert not p.exists()
