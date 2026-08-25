# syntax=docker/dockerfile:1
#
# roqsim (lean core) — ROS-free, headless MuJoCo image.
#
# Multi-stage so pip caches / build tooling never reach the final image. Layers are ordered
# stable -> volatile: third-party runtime deps are installed first (rarely change), the roqsim_*
# sources are copied and installed last (change often), so editing our code only rebuilds the cheap
# final --no-deps install layer.
#
# Contains: core + assets + sensors + mobile + manipulation + scenes + walker + rso.
# NOT included: quadruped/humanoid (they pull torch). Set INCLUDE_TORCH_PKGS=1 for a full non-ROS
# variant (adds CPU-only torch + quadruped + humanoid).

ARG PYTHON_VERSION=3.12

##############################################################
####                      builder                         ####
##############################################################
FROM python:${PYTHON_VERSION}-slim AS builder
SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

ARG INCLUDE_TORCH_PKGS=0

# All installs go into a self-contained venv we copy into the final stage.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /build

# 1) Stable layer: third-party runtime deps (pinned loosely, mirrors the sub-package pyproject deps).
RUN pip install --no-cache-dir \
        "mujoco>=3.0" \
        "numpy>=1.24" \
        "pyyaml>=6.0" \
        "click>=8.1" \
        "pillow>=9.0" \
        "lxml>=4.0" \
        "py_trees>=2.2" \
        "trimesh>=3.20" \
        "scipy>=1.11" \
        "requests>=2.25" \
        "opencv-python-headless>=4.6"

# 1b) Optional torch-backed packages for a full non-ROS build. CPU wheel index only (no CUDA);
#     kept in its own guarded layer so the lean default never pays for it.
RUN if [ "${INCLUDE_TORCH_PKGS}" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir "onnxruntime>=1.16"; \
    fi

# 2) Volatile layer: our package sources (each dir carries its own pyproject.toml + package data).
COPY roqsim            ./roqsim
COPY roqsim_assets     ./roqsim_assets
COPY roqsim_sensors    ./roqsim_sensors
COPY roqsim_mobile     ./roqsim_mobile
COPY roqsim_manipulation ./roqsim_manipulation
COPY roqsim_scenes     ./roqsim_scenes
COPY roqsim_walker     ./roqsim_walker
COPY roqsim_webctrl        ./roqsim_webctrl

# 3) Install our packages with deps already satisfied above (--no-deps keeps this layer cheap and
#    makes a source edit rebuild only this step). roqsim (core) first, then the dependents.
RUN pip install --no-cache-dir --no-deps \
        ./roqsim \
        ./roqsim_assets \
        ./roqsim_sensors \
        ./roqsim_mobile \
        ./roqsim_manipulation \
        ./roqsim_scenes \
        ./roqsim_walker \
        ./roqsim_webctrl

# 3b) Optional torch-backed packages (source + install) for the full non-ROS variant.
COPY roqsim_quadruped  ./roqsim_quadruped
COPY roqsim_humanoid   ./roqsim_humanoid
RUN if [ "${INCLUDE_TORCH_PKGS}" = "1" ]; then \
        pip install --no-cache-dir --no-deps ./roqsim_quadruped ./roqsim_humanoid; \
    fi

# 4) Every --no-deps install above trusted the hand-written list in step 1 to already carry what the
#    pyprojects declare. Nothing checked that, and it had drifted: `click` was absent, so this
#    image's own ENTRYPOINT (`roqsim`, with CMD `--help`) exited with ModuleNotFoundError. Building
#    fine and failing on first run is the whole failure mode -- so the drift is loud here now.
COPY container/files/check_runtime_deps.py /tmp/check_runtime_deps.py
RUN python /tmp/check_runtime_deps.py && rm /tmp/check_runtime_deps.py

##############################################################
####                       final                          ####
##############################################################
FROM python:${PYTHON_VERSION}-slim AS final
SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

# Runtime system libs for headless MuJoCo. BOTH offscreen backends, because which one works is a
# property of the node this image lands on, not of the image: egl needs a DRI render device, and a
# CPU-only node has none. roqsim picks between them at import (roqsim.gl.select_offscreen_gl), so
# shipping only one is what turns "no GPU here" into an unexplained render abort.
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libgl1-mesa-dri \
        libosmesa6 \
        libglib2.0-0 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# MUJOCO_GL is deliberately NOT set. Baking egl here would win over roqsim's own selection (an
# explicit value is always honoured) and hand a CPU-only node a backend it cannot initialise, with
# no fallback -- the image would be pinning a decision only the node can make.
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1

# Run as a non-root user.
RUN useradd --create-home --uid 1000 simsuite
USER simsuite
WORKDIR /home/simsuite

# The `roqsim` console script (roqsim.runner:main). Pass e.g. `<w> --headless ...`.
ENTRYPOINT ["roqsim"]
CMD ["--help"]
