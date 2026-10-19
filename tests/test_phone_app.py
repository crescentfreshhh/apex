"""The phone app: Chrome can install Peaks (manifest, service worker, icons —
reachable even behind the password gate), and a sign-in lasts: signed
sessions survive a restart, "keep me signed in" lasts 90 days, a password
change or sign-out ends them, and an active session slides on."""

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks.config import Config  # noqa: E402
from peaks.web.app import create_app  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.embedding.cache_dir = str(tmp_path / "cache")
    c.modeling.dir = str(tmp_path / "models")
    c.auth.password = "hunter2"
    c.auth.session_hours = 1.0
    return c


def _login(client, remember=False):
    r = client.post("/api/login", json={"password": "hunter2", "remember": remember})
    assert r.status_code == 200
    return r


def test_manifest_describes_an_installable_app(cfg):
    m = TestClient(create_app(cfg)).get("/manifest.webmanifest")
    assert m.status_code == 200 and m.headers["content-type"].startswith("application/manifest+json")
    j = m.json()
    assert j["display"] == "standalone" and j["start_url"] == "/" and j["scope"] == "/"
    sizes = {(i["sizes"], i.get("purpose", "any")) for i in j["icons"]}
    assert {("192x192", "any"), ("512x512", "any"), ("512x512", "maskable")} <= sizes


def test_install_bits_pass_the_gate_but_the_app_does_not(cfg):
    c = TestClient(create_app(cfg))
    assert c.get("/manifest.webmanifest").status_code == 200
    sw = c.get("/sw.js")
    assert sw.status_code == 200 and "javascript" in sw.headers["content-type"] and "/api/" in sw.text
    for icon in ("icon-192.png", "icon-512.png", "maskable-512.png", "apple-touch-icon.png"):
        r = c.get("/static/icons/" + icon)
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert "sign in" in c.get("/").text.lower()                      # the app itself: login page
    assert "sign in" in c.get("/static/app.js").text.lower()
    assert c.get("/api/stats").status_code == 401


def test_a_session_survives_a_restart(cfg):
    a = TestClient(create_app(cfg))
    _login(a)
    b = TestClient(create_app(cfg))                                  # a new process (Docker pull)
    b.cookies.set("peaks_session", a.cookies.get("peaks_session"))
    assert b.get("/api/stats").status_code == 200


def test_password_change_ends_every_session(cfg):
    a = TestClient(create_app(cfg))
    _login(a)
    cfg.auth.password = "new-one"
    b = TestClient(create_app(cfg))
    b.cookies.set("peaks_session", a.cookies.get("peaks_session"))
    assert b.get("/api/stats").status_code == 401


def test_keep_me_signed_in_lasts_90_days(cfg):
    c = TestClient(create_app(cfg))
    assert "max-age=3600" in _login(c).headers["set-cookie"].lower()
    assert f"max-age={90 * 86400}" in _login(c, remember=True).headers["set-cookie"].lower()


def test_expiry_tampering_and_sign_out(cfg, monkeypatch):
    c = TestClient(create_app(cfg))
    _login(c)
    tok = c.cookies.get("peaks_session")
    exp, ttl, nonce, sig = tok.split(".")
    c.cookies.set("peaks_session", f"{int(exp) + 10**6}.{ttl}.{nonce}.{sig}")   # a longer expiry: forged
    assert c.get("/api/stats").status_code == 401
    c.cookies.set("peaks_session", tok)
    assert c.get("/api/stats").status_code == 200
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 2 * 3600)             # two hours idle: expired
    assert c.get("/api/stats").status_code == 401
    monkeypatch.setattr(time, "time", real)
    c.post("/api/logout")
    c.cookies.set("peaks_session", tok)                                       # the signed-out token
    assert c.get("/api/stats").status_code == 401


def test_an_active_session_slides_on(cfg, monkeypatch):
    c = TestClient(create_app(cfg))
    _login(c)
    assert "set-cookie" not in c.get("/api/stats").headers                   # fresh: nothing to renew
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 40 * 60)               # past halfway
    r = c.get("/api/stats")
    assert r.status_code == 200 and "max-age=3600" in r.headers["set-cookie"].lower()


def test_every_page_file_ships_in_the_package():
    """The Docker image installs the package (no git, no build leftovers), so a
    page file the package-data globs miss is missing on the server — that's how
    the app icons went missing and Chrome refused to install Peaks."""
    import fnmatch
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    globs = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["peaks.web"]
    static = root / "src" / "peaks" / "web" / "static"
    files = [p.relative_to(static.parent).as_posix() for p in static.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    missed = [f for f in files if not any(fnmatch.fnmatch(f, g) and f.count("/") == g.count("/") for g in globs)]
    assert files and not missed, f"not in the package: {missed}"
