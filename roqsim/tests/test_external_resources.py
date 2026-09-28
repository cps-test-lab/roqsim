"""``external/external_resources.py`` fetches a source whole or not at all.

A source that declares no ``sha256`` is told apart from a missing one only by existing, so a transfer
that broke off part-way must leave nothing behind: otherwise the next run reports ``have`` and the
conversion runs on a truncated file. Skipped in a checkout that does not carry ``external/``.
"""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "external" / "external_resources.py"
if not _SCRIPT.exists():
    pytest.skip("external/ is not part of this checkout", allow_module_level=True)

_SOURCE = "external/sources/vendor/body.stl"


@pytest.fixture
def ext(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("external_resources", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def _breaks_off(url, filename):
    """A transfer that wrote part of the file and then lost the connection."""
    Path(filename).write_bytes(b"half of a mesh")
    raise urllib.error.ContentTooShortError("retrieval incomplete", b"")


def _completes(url, filename):
    Path(filename).write_bytes(b"the whole mesh")


def test_a_broken_transfer_leaves_nothing_at_the_path(ext, tmp_path, monkeypatch):
    monkeypatch.setattr(ext.urllib.request, "urlretrieve", _breaks_off)
    resource = {
        "name": "vendor_meshes",
        "sources": [{"url": "https://example.com/b.stl", "path": _SOURCE}],
    }
    with pytest.raises(SystemExit):
        ext._fetch(resource)
    assert not (tmp_path / _SOURCE).exists()
    assert list((tmp_path / _SOURCE).parent.iterdir()) == []


def test_an_optional_resource_is_fetched_again_after_a_broken_transfer(
    ext, tmp_path, monkeypatch, capsys
):
    resource = {
        "name": "vendor_meshes",
        "optional": True,
        "sources": [{"url": "https://example.com/b.stl", "path": _SOURCE}],
    }
    monkeypatch.setattr(ext.urllib.request, "urlretrieve", _breaks_off)
    assert ext._fetch(resource) is False
    monkeypatch.setattr(ext.urllib.request, "urlretrieve", _completes)
    assert ext._fetch(resource) is True
    assert (tmp_path / _SOURCE).read_bytes() == b"the whole mesh"
    assert "have" not in capsys.readouterr().out


def test_a_whole_file_is_kept_and_not_fetched_again(ext, tmp_path, monkeypatch, capsys):
    resource = {
        "name": "vendor_meshes",
        "sources": [{"url": "https://example.com/b.stl", "path": _SOURCE}],
    }
    monkeypatch.setattr(ext.urllib.request, "urlretrieve", _completes)
    assert ext._fetch(resource) is True
    monkeypatch.setattr(ext.urllib.request, "urlretrieve", _breaks_off)
    assert ext._fetch(resource) is True
    assert (tmp_path / _SOURCE).read_bytes() == b"the whole mesh"
    assert "have" in capsys.readouterr().out


def test_a_failed_conversion_leaves_no_target(ext, tmp_path, monkeypatch):
    target = "roqsim_sensors/meshes/body.obj"
    resource = {
        "name": "vendor_meshes",
        "sources": [{"url": "https://example.com/b.stl", "path": _SOURCE}],
        "convert": {"script": "external/convert/body.py"},
        "targets": [target],
    }
    monkeypatch.setattr(ext.urllib.request, "urlretrieve", _completes)
    monkeypatch.setattr(ext, "_toolvenv_python", lambda: Path("python"))

    def _stops_half_way(cmd, check):
        out = tmp_path / target
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"half of a mesh")
        raise ext.subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(ext.subprocess, "run", _stops_half_way)
    with pytest.raises(ext.subprocess.CalledProcessError):
        ext._convert(resource)
    assert not (tmp_path / target).exists()
