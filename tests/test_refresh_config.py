"""Config refresh logic (docker/refresh_config.py) — seeding, up-to-date
no-op, and the stale-refresh that backs up + preserves [stash]."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "refresh_config",
    Path(__file__).resolve().parents[1] / "docker" / "refresh_config.py",
)
refresh_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh_config)
refresh = refresh_config.refresh


EXAMPLE_V1 = """[stash]
url = "http://192.168.1.2:6969"
api_key = ""
timeout = 30

[scoring]
max_duration = 30.0
"""

EXAMPLE_V2 = """[stash]
url = "http://192.168.1.2:6969"
api_key = ""
timeout = 30

[scoring]
max_duration = 30.0
high = 0.55
"""


def _paths(tmp_path):
    ex = tmp_path / "config.example.toml"
    cfg = tmp_path / "config.toml"
    ver = tmp_path / ".config_version"
    return ex, cfg, ver


def test_seeds_when_missing(tmp_path):
    ex, cfg, ver = _paths(tmp_path)
    ex.write_text(EXAMPLE_V1)
    assert refresh(ex, cfg, ver, log=lambda *_: None) == "seeded"
    assert cfg.read_text() == EXAMPLE_V1
    assert ver.read_text()  # fingerprint stored


def test_noop_when_unchanged(tmp_path):
    ex, cfg, ver = _paths(tmp_path)
    ex.write_text(EXAMPLE_V1)
    refresh(ex, cfg, ver, log=lambda *_: None)
    # user tweaks their file; example is unchanged → must NOT be touched
    cfg.write_text(EXAMPLE_V1 + "\n# my notes\n")
    before = cfg.read_text()
    assert refresh(ex, cfg, ver, log=lambda *_: None) == "current"
    assert cfg.read_text() == before


def test_refresh_backs_up_and_preserves_stash(tmp_path):
    ex, cfg, ver = _paths(tmp_path)
    ex.write_text(EXAMPLE_V1)
    refresh(ex, cfg, ver, log=lambda *_: None)

    # user had a stale scoring value AND a real api key in config.toml
    cfg.write_text(
        '[stash]\nurl = "http://10.0.0.9:9999"\napi_key = "SECRET123"\ntimeout = 45\n\n'
        "[scoring]\nmax_duration = 0.0\n"
    )
    # ship new defaults
    ex.write_text(EXAMPLE_V2)
    action = refresh(ex, cfg, ver, now="TS", log=lambda *_: None)
    assert action == "refreshed"

    new = cfg.read_text()
    # tuning was refreshed (stale max_duration=0 gone, new default present)
    assert "max_duration = 30.0" in new and "max_duration = 0.0" not in new
    assert "high = 0.55" in new
    # connection block carried across
    assert 'url = "http://10.0.0.9:9999"' in new
    assert 'api_key = "SECRET123"' in new
    assert "timeout = 45" in new
    # old file backed up verbatim
    backup = tmp_path / "config.toml.bak.TS"
    assert backup.exists() and "max_duration = 0.0" in backup.read_text()


def test_missing_example_is_noop(tmp_path):
    ex, cfg, ver = _paths(tmp_path)
    assert refresh(ex, cfg, ver, log=lambda *_: None) == "current"
    assert not cfg.exists()
