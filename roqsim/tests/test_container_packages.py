"""The container images carry every roqsim package but the ones ``container/packages.py`` excludes.

The images build from that script rather than from a list, and check at build time that exactly its
set got installed. These tests hold the script to the repository: it finds what the Makefile finds,
each exclusion names a real package, and no Dockerfile names a package of its own.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCKERFILES = {"Dockerfile": "lean", "Dockerfile.ros": "ros"}


@pytest.fixture(scope="module")
def packages():
    pytest.importorskip("tomllib")  # stdlib from 3.11; the images run 3.12
    spec = importlib.util.spec_from_file_location("image_packages", REPO / "container/packages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_images_find_the_packages_the_makefile_does(packages):
    out = subprocess.run(
        [
            "make",
            "-s",
            "--no-print-directory",
            "--eval",
            "print-pkgs: ; @echo $(PKGS)",
            "print-pkgs",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    ).stdout.split()
    assert list(packages.discover()) == sorted(out)


@pytest.mark.parametrize("image", ["lean", "ros"])
def test_an_image_carries_every_package_but_its_named_exclusions(packages, image):
    found = set(packages.discover())
    excluded = packages.EXCLUDE[image]
    assert set(excluded) <= found, f"{image} excludes a package that does not exist"
    assert all(reason.strip() for reason in excluded.values())
    assert set(packages.for_image(image)) == found - set(excluded)


def test_the_lean_image_leaves_out_exactly_the_packages_that_need_torch(packages):
    needs_torch = {name for name, p in packages.discover().items() if packages.requires(p, "torch")}
    excluded = {
        name for name, reason in packages.EXCLUDE["lean"].items() if reason == packages.TORCH
    }
    assert excluded == needs_torch


@pytest.mark.parametrize("dockerfile,image", DOCKERFILES.items())
def test_a_dockerfile_installs_the_discovered_set_and_names_no_package(dockerfile, image):
    text = (REPO / "container" / dockerfile).read_text()
    assert f"container/packages.py {image})" in text
    assert f"container/packages.py {image} --check" in text
    listed = re.findall(r"(?:\./|COPY\s+)((?:roqsim|scenario_execution_)\w*)", text)
    assert not listed, f"{dockerfile} names packages by hand: {listed}"


def test_the_build_time_check_refuses_a_package_left_out(packages, monkeypatch, capsys):
    carried = packages.for_image("ros")
    projects = packages.discover()
    installed = [SimpleNamespace(metadata={"Name": projects[n]["name"]}) for n in carried[1:]]
    monkeypatch.setattr(packages.md, "distributions", lambda: installed)
    assert packages.check("ros") == 1
    assert carried[0] in capsys.readouterr().err.replace("-", "_")
