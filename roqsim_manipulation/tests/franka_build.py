"""Build the real libfranka at the pinned release, and the test client against it, once per machine.

``test_franka_libfranka.py`` runs Franka's own client library against the simulated FCI: that is the
in-sync test of the protocol, the libfranka-side rate limiter and filter, and ``Model``. libfranka is
cloned at the tag ``SOURCES.toml`` pins (its commit is checked), built with the C++ toolchain and
its dependencies -- Poco and Eigen from the system, Pinocchio, urdfdom and tinyxml2 from the ``cmeel``
Python packages (``pin`` and ``cmeel-urdfdom-headers``, the ``test`` extra) -- and cached under ``ROQSIM_FRANKA_CACHE``
(default ``~/.cache/roqsim/franka``). A failure is a test failure naming the step, never a skip.
``make test-franka`` runs it; ``make test`` leaves it out, for the C++ toolchain it needs.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from _franka_upstream import sources

from roqsim import env

HERE = Path(__file__).resolve().parent


class BuildError(RuntimeError):
    pass


def _libfranka_pin() -> dict:
    return next(r for r in sources()["repo"] if r["dir"] == "libfranka")


def _cmeel_prefixes() -> list[Path]:
    out = []
    for p in sys.path:
        prefix = Path(p) / "cmeel.prefix"
        if prefix.is_dir() and prefix not in out:
            out.append(prefix)
    return out


def _run(cmd: list[str], cwd: Path, log: Path, env: dict | None = None) -> None:
    with log.open("a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        tail = log.read_text(errors="replace")[-3000:]
        raise BuildError(f"{' '.join(cmd[:3])} failed (log {log}):\n{tail}")


def runtime_env() -> dict:
    env = dict(os.environ)
    libs = [str(p / "lib") for p in _cmeel_prefixes()] + [str(_cache() / "install" / "lib")]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    return env


def _cache() -> Path:
    pin = _libfranka_pin()
    root = Path(env.get("ROQSIM_FRANKA_CACHE") or Path.home() / ".cache" / "roqsim" / "franka")
    return root / pin["commit"][:12]


def build_client() -> Path:
    """The ``fci_client`` binary, linked against libfranka at the pinned commit."""
    pin = _libfranka_pin()
    cache = _cache()
    cache.mkdir(parents=True, exist_ok=True)
    log = cache / "build.log"
    src, build, install = cache / "src", cache / "build", cache / "install"
    prefixes = ";".join([str(install)] + [str(p) for p in _cmeel_prefixes()])
    if not (install / "lib" / "libfranka.so").exists():
        if not (src / ".git").exists():
            _run(
                [
                    "git",
                    "clone",
                    "-q",
                    "--depth",
                    "1",
                    "--branch",
                    pin["tag"],
                    "--recurse-submodules",
                    "--shallow-submodules",
                    pin["url"],
                    str(src),
                ],
                cache,
                log,
            )
        head = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        if head != pin["commit"]:
            raise BuildError(
                f"libfranka tag {pin['tag']} is {head}, SOURCES.toml pins {pin['commit']}"
            )
        build.mkdir(exist_ok=True)
        _run(
            [
                "cmake",
                str(src),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_TESTS=OFF",
                "-DBUILD_EXAMPLES=OFF",
                f"-DCMAKE_PREFIX_PATH={prefixes}",
                f"-DCMAKE_INSTALL_PREFIX={install}",
            ],
            build,
            log,
        )
        _run(["cmake", "--build", ".", "-j", str(os.cpu_count() or 2)], build, log)
        _run(["cmake", "--install", "."], build, log)
    # One client build per checkout: CMake ties a build directory to the source it configured.
    client_build = cache / ("client-" + hashlib.sha256(str(HERE).encode()).hexdigest()[:12])
    client = client_build / "fci_client"
    source = HERE / "franka_clients" / "fci_client.cpp"
    if not client.exists() or client.stat().st_mtime < source.stat().st_mtime:
        client_build.mkdir(exist_ok=True)
        rpath = ":".join(str(p / "lib") for p in _cmeel_prefixes())
        _run(
            [
                "cmake",
                str(HERE / "franka_clients"),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_PREFIX_PATH={prefixes}",
                f"-Dfmt_DIR={build / '_deps' / 'fmt-build'}",
                f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{rpath}",
            ],
            client_build,
            log,
        )
        _run(["cmake", "--build", "."], client_build, log)
    return client
