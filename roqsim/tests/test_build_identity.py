# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The build identity: which commit an installed roqsim came from.

The version names a release line that every build between two releases shares, so two images that
behave differently cannot be told apart by it. What these tests hold:

- a regular build bakes the commit, taken from ``ROQSIM_GIT_SHA`` or from git, and a build that can
  name neither FAILS rather than baking a placeholder;
- an editable install reads git each time, and only for a tree that really holds roqsim;
- an identity that is absent is reported as absent, with its reason, never guessed;
- ``roqsim --version`` and both introspection answers carry it.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from roqsim import build_identity as bi

SHA = "0123456789abcdef0123456789abcdef01234567"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # roqsim/ -- the directory with setup.py


def _git(cwd, *args):
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    )


def _repo(tmp_path):
    """A one-commit repository holding an ``__init__.py``; returns ``(root, commit)``."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "__init__.py").write_text("")
    _git(root, "init", "-q")
    _git(root, "add", "__init__.py")
    _git(root, "commit", "-q", "-m", "init")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return root, head


# -- the value itself ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [SHA, f"{SHA}-dirty", f"  {SHA}\n"])
def test_a_full_commit_is_a_build(text):
    commit, dirty = bi.parse_build(text, origin="t")
    assert commit == SHA
    assert dirty == text.strip().endswith("-dirty")


@pytest.mark.parametrize("text", ["", "unknown", SHA[:12], SHA.upper(), f"{SHA}-modified"])
def test_anything_else_is_refused_and_says_where_it_came_from(text):
    """An abbreviation is refused too: it is ambiguous as a repository grows, and this value is kept
    next to results for as long as they are."""
    with pytest.raises(bi.BuildIdentityError, match="ROQSIM_GIT_SHA"):
        bi.parse_build(text, origin="ROQSIM_GIT_SHA")


# -- what a build bakes -------------------------------------------------------------------------


def test_the_explicit_commit_wins_over_git(tmp_path, monkeypatch):
    root, head = _repo(tmp_path)
    monkeypatch.setenv(bi.ENV_VAR, f"{SHA}-dirty")
    assert bi.for_build(root) == {"commit": SHA, "dirty": True}
    assert head != SHA


def test_a_malformed_explicit_commit_is_refused_not_passed_over_for_git(tmp_path, monkeypatch):
    root, _ = _repo(tmp_path)
    monkeypatch.setenv(bi.ENV_VAR, "latest")
    with pytest.raises(bi.BuildIdentityError, match="not a build identity"):
        bi.for_build(root)


def test_without_the_variable_git_answers_and_notices_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.delenv(bi.ENV_VAR, raising=False)
    root, head = _repo(tmp_path)
    assert bi.for_build(root) == {"commit": head, "dirty": False}
    (root / "__init__.py").write_text("changed = True\n")
    assert bi.for_build(root) == {"commit": head, "dirty": True}


def test_an_untracked_file_does_not_make_a_tree_dirty(tmp_path, monkeypatch):
    """A build directory or a cache in the tree says nothing about the sources installed."""
    monkeypatch.delenv(bi.ENV_VAR, raising=False)
    root, head = _repo(tmp_path)
    (root / "build").mkdir()
    (root / "build" / "leftover.txt").write_text("x")
    assert bi.for_build(root) == {"commit": head, "dirty": False}


def test_a_build_that_cannot_name_its_commit_fails(tmp_path, monkeypatch):
    """The whole point: no ``unknown`` baked into an image that then looks identified."""
    monkeypatch.delenv(bi.ENV_VAR, raising=False)
    with pytest.raises(bi.BuildIdentityError, match="ROQSIM_GIT_SHA"):
        bi.for_build(tmp_path)


# -- what an installation reports ---------------------------------------------------------------


def test_a_baked_identity_is_read_first(tmp_path, monkeypatch):
    (tmp_path / bi.BAKED_FILE).write_text(json.dumps({"commit": SHA, "dirty": False}))
    monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path)
    identity = bi.build_identity()
    assert identity == bi.BuildIdentity(commit=SHA, dirty=False, source="baked")
    assert bi.version_line().endswith(f", build {SHA}")


def test_a_corrupt_baked_identity_is_an_error_not_an_absence(tmp_path, monkeypatch):
    (tmp_path / bi.BAKED_FILE).write_text(json.dumps({"commit": "unknown"}))
    monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path)
    with pytest.raises(bi.BuildIdentityError):
        bi.build_identity()


def test_a_checkout_is_asked_each_time(tmp_path, monkeypatch):
    root, head = _repo(tmp_path)
    monkeypatch.setattr(bi, "_PACKAGE_DIR", root)
    assert bi.build_identity() == bi.BuildIdentity(commit=head, dirty=False, source="checkout")
    (root / "__init__.py").write_text("changed = True\n")
    assert bi.version_line().endswith(f", build {head}-dirty")


def test_a_directory_merely_inside_a_repository_is_not_its_checkout(tmp_path, monkeypatch):
    """A virtualenv kept in a repository sits inside that repository's work tree; a package
    installed there must not report the repository's commit as its own."""
    root, _ = _repo(tmp_path)
    site = root / ".venv" / "site-packages" / "roqsim"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("")
    monkeypatch.setattr(bi, "_PACKAGE_DIR", site)
    assert bi.build_identity() is None


def test_an_absent_identity_is_stated_with_its_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path)
    described = bi.describe()
    assert described["commit"] is None
    assert bi.BAKED_FILE in described["reason"]
    assert ", build not recorded (" in bi.version_line()


def test_describe_carries_the_version_and_the_identity(tmp_path, monkeypatch):
    import roqsim

    (tmp_path / bi.BAKED_FILE).write_text(json.dumps({"commit": SHA, "dirty": True}))
    monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path)
    assert bi.describe() == {
        "version": roqsim.__version__,
        "commit": SHA,
        "dirty": True,
        "source": "baked",
    }


# -- where it is exposed ------------------------------------------------------------------------


def test_roqsim_version_prints_the_build(tmp_path, monkeypatch):
    from roqsim.commands import cli

    (tmp_path / bi.BAKED_FILE).write_text(json.dumps({"commit": SHA, "dirty": False}))
    monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path)
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    import roqsim

    assert result.output.strip() == f"roqsim, version {roqsim.__version__}, build {SHA}"


def test_both_introspection_answers_name_the_build_that_gave_them(tmp_path, monkeypatch):
    from roqsim import introspection

    (tmp_path / bi.BAKED_FILE).write_text(json.dumps({"commit": SHA, "dirty": False}))
    monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path)
    assert introspection.list_plugins()["roqsim"]["commit"] == SHA
    assert introspection.get_plugin_details("dummy")["roqsim"]["commit"] == SHA


# -- the build step itself ----------------------------------------------------------------------


def _wheel(source, out, env):
    return subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "--no-build-isolation"]
        + ["-w", str(out), str(source)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def test_a_wheel_built_without_git_carries_the_commit_it_was_given(tmp_path):
    """A container build has no ``.git``: the commit arrives as ``ROQSIM_GIT_SHA``, and without it
    the build fails instead of producing an install that cannot name itself."""
    source = tmp_path / "roqsim"
    shutil.copytree(
        PACKAGE_ROOT,
        source,
        ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", ".pytest_cache"),
    )
    env = {k: v for k, v in os.environ.items() if k != bi.ENV_VAR}
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)  # the copy must not find an enclosing repo

    refused = _wheel(source, tmp_path / "refused", env)
    assert refused.returncode != 0
    assert "BuildIdentityError" in refused.stdout + refused.stderr

    shutil.rmtree(source / "build", ignore_errors=True)
    built = _wheel(source, tmp_path / "wheel", {**env, bi.ENV_VAR: SHA})
    assert built.returncode == 0, built.stdout + built.stderr
    (wheel,) = (tmp_path / "wheel").glob("roqsim-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        baked = json.loads(archive.read(f"roqsim/{bi.BAKED_FILE}"))
    assert baked == {"commit": SHA, "dirty": False}
