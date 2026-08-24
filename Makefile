# roqsim developer workflow.
#
# The venv is created with --system-site-packages so a *sourced* ROS 2 (rclpy, nav2,
# simulation_interfaces) is importable by the same interpreter that has mujoco + our editable
# packages. This avoids the "ros2 run uses /usr/bin/python3 which lacks roqsim" footgun.
#
# Typical flow:
#   make venv                         # one-time: create .venv, install packages + tooling
#   source /opt/ros/jazzy/setup.bash  # (only needed for ROS / nav2)
#   make test                         # unit tests always; + nav2 integration when ROS is sourced

VENV       ?= .venv
PY         := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
# Invoke tools as modules: with a --system-site-packages venv, console scripts (pytest, ruff,
# sphinx-build) may be satisfied from the system and have no wrapper in $(VENV)/bin.
# Third-party pytest plugins are never wanted here, and with a sourced ROS they are actively harmful:
# --system-site-packages makes ROS's launch_testing / launch_testing_ros entry points visible, and they
# declare hooks with signatures modern pytest removed, so collection dies with PluginValidationError
# before a single test runs. Nothing in this tree uses a plugin -- fixtures, monkeypatch, parametrize
# and skipif are all core -- so autoload goes off rather than blocking each offender by name (the
# first attempt did that and found a second plugin behind the first).
export PYTEST_DISABLE_PLUGIN_AUTOLOAD ?= 1
PYTEST     := $(PY) -m pytest
RUFF       := $(PY) -m ruff
SPHINX     := $(PY) -m sphinx
# MUJOCO_GL is deliberately NOT set here. `roqsim.gl.select_offscreen_gl` chooses it per MACHINE at
# import time -- egl where /dev/dri/renderD128 exists, osmesa where it does not -- and an explicit
# value is always honoured, so exporting one here overrode that decision for the whole suite with a
# guess that is simply wrong on a CPU-only host. It also cost two CI fixes: a GPU-less runner needed
# libosmesa6 installed anyway, and test_gl_boot had to stop inheriting the PYOPENGL_PLATFORM that
# `import mujoco` derives from this variable. Set it in your own shell to override; do not put it back.
# Blender binary used by external-resource conversions that need it (e.g. the Livox meshes). Override
# with `make external-resources BLENDER=/path/to/blender` if it is not on PATH.
export BLENDER ?= blender
EXTERNAL   := $(PY) external/external_resources.py

# Every package in this dir (each has a pyproject.toml), discovered so a new sibling package is
# picked up by `make venv`/`make test`/`make lint` with no edit here -- as long as it matches one of
# the two name shapes: `roqsim*` for the substrate's own, `scenario_execution_*` for a scenario-execution
# action library, which follows THAT project's naming convention (its own libs are all
# `scenario_execution_<thing>`) rather than ours. sort puts the core `roqsim` first (shorter name sorts
# first) and dedups.
PKGS       := $(sort $(patsubst %/,%,$(dir \
	$(wildcard roqsim*/pyproject.toml) $(wildcard scenario_execution_*/pyproject.toml))))
SRC        := $(addsuffix /src,$(PKGS)) ros2_ws/src
# Tests that need ROS on the path, so they run only in the sourced branch of `make test`. Every
# colcon package's own test/ dir, discovered the same way PKGS is -- the bridge's suite had been
# sitting outside `make test` entirely because only the nav2 example was named here.
ROS_TESTS  := $(wildcard ros2_ws/src/*/test)
# Every world any package ships, for `make smoke`. Globbed rather than asked of the CLI: there is no
# worlds-listing subcommand, and a glob also catches a world a package forgot to register.
#
# Split the same way ROS_TESTS is, and for the same reason: a ros2_ws world declares the `ros2_bridge`
# / `sim_interfaces` transport plugins, which ship in a colcon package and cannot resolve in a
# pip-only environment. Those run with --no-communication when ROS is not sourced -- the world still
# compiles and every mesh and texture still has to resolve, which is what smoke is actually for, it
# just publishes nothing. With ROS sourced they run as authored.
WORLDS     := $(wildcard $(addsuffix /src/*/worlds/*.yaml,$(PKGS)))
ROS_WORLDS := $(wildcard ros2_ws/src/*/worlds/*.yaml)
SMOKE_STEPS ?= 200

# pip extras per package, looked up as EXTRAS_<pkg> with EXTRAS_DEFAULT as the fallback. Deliberately
# a lookup rather than $(if $(filter ...)): make splits a function's arguments on commas before
# expanding them, so an extras list written literally inside $(if ...) is torn into separate
# arguments -- which is what made `make venv` fail with `-e "rstmarkers,coverage],[test]"`.
# Giving another package extras is now one line here and no change below.
#
# These keys MUST match the directory names in $(PKGS). One was left at its pre-rename name after the
# rst -> roqsim rename and silently stopped matching: roqsim_sensors fell back to [test], so
# matplotlib never entered the venv and the NumPy-1-built SYSTEM matplotlib leaked in through
# --system-site-packages instead, failing the coverage and passability tests with
# "numpy.core.multiarray failed to import". Same failure mode the scipy pin in roqsim_walker
# documents. If a package's tests import matplotlib, its extras belong here.
EXTRAS_DEFAULT        := [test]
EXTRAS_roqsim_sensors := [test,markers,coverage]
EXTRAS_roqsim_scenes  := [test,preview]

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv:  ## Create .venv (--system-site-packages) and install packages + dev tooling
	python3 -m venv --system-site-packages $(VENV)
	$(PIP) install --upgrade pip
	# Install PyTorch from the CPU wheel index first. roqsim_quadruped/roqsim_humanoid depend on
	# torch, and the default PyPI wheel drags in multi-GB CUDA libraries this MuJoCo/EGL sim never uses
	# (rendering is GL, policies run fine on CPU). Preinstalling the CPU build satisfies that dependency
	# so the editable installs below stay lean -- and so every models package (incl. spot / unitree_g1)
	# actually installs and shows up in the docs model catalog.
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	# roqsim_sensors also gets its markers (fiducials) + coverage (matplotlib heatmap) extras; see
	# EXTRAS_<pkg> above for why this is a lookup and not an inline conditional.
	$(PIP) install $(foreach p,$(PKGS),-e "$(p)$(or $(EXTRAS_$(p)),$(EXTRAS_DEFAULT))")
	# --ignore-installed for pytest, not tidiness: with --system-site-packages pip counts the SYSTEM
	# pytest as satisfying `pytest>=7.0` from every package's [test] extra, so the runner is never
	# installed here and whatever the host or base image ships is what runs. That is invisible until
	# the host changes -- inside ros:jazzy its pytest collected 1 test instead of 1449 and `make test`
	# exited 5 ("no tests collected") while the same commit passed everywhere else. The venv owns its
	# own test runner.
	$(PIP) install --ignore-installed pytest
	# ruff is PINNED because it is the formatter: `ruff format` output changes between releases,
	# so an unpinned one turns `make lint` red on a commit that touched nothing -- 0.16.4 rewrapped
	# three files 0.16.3 was happy with, and main went red with no code change behind it. It also
	# breaks what the CI workflow promises, that a contributor reproduces CI with one make target:
	# a contributor whose ruff differs from CI's cannot. Bump this deliberately, reformatting in
	# the same commit. sphinx/furo only render docs, so they stay floating.
	$(PIP) install sphinx furo ruff==0.16.4
	$(EXTERNAL) convert --resource spot_locomotion_policy  # fetch the NVIDIA Spot policy (external asset; fail-soft)
	@echo
	@echo "venv ready. For ROS/nav2: 'source /opt/ros/jazzy/setup.bash' then 'make test'."
	@echo "nav2 demos: 'ros2 launch roqsim_nav2_example nav2_g1.launch.py' (or nav2_spot / nav2_turtlebot)."

.PHONY: build-ros
build-ros:  ## colcon build ros2_ws (only if ROS is sourced)
	@if [ -n "$$ROS_DISTRO" ]; then \
		echo "colcon build (ROS_DISTRO=$$ROS_DISTRO) ..."; \
		cd ros2_ws && colcon build; \
	else \
		echo "ROS not sourced; skipping colcon build."; \
	fi

.PHONY: test
test:  ## Run unit tests; also the ros2_ws tests (incl. nav2 integration) when ROS is sourced
	$(PYTEST) $(PKGS) -q
	@if [ -n "$$ROS_DISTRO" ]; then \
		echo "ROS sourced -> building ros2_ws and running its tests"; \
		$(MAKE) build-ros; \
		. ros2_ws/install/setup.sh && $(PYTEST) $(ROS_TESTS) -q; \
	else \
		echo "ROS not sourced -> skipped the ros2_ws tests (source ROS to include them)."; \
	fi

.PHONY: smoke
smoke:  ## Headless-run every shipped world (compiles the MJCF, loads plugins, resolves assets)
	@fail=""; skipped=""; n=0; log=$$(mktemp); \
	if [ -n "$$ROS_DISTRO" ] && [ -f ros2_ws/install/setup.sh ]; then mute=""; note="as authored (ROS sourced)"; \
	else mute="--no-communication"; note="with --no-communication (ROS not sourced)"; fi; \
	run() { \
		n=$$((n+1)); printf '== %s\n' "$$1"; \
		if $(VENV)/bin/roqsim sim "$$1" --headless --pacing asap --steps $(SMOKE_STEPS) $$2 >"$$log" 2>&1; then return 0; fi; \
		miss=$$(sed -n "s/.*Error opening file '\([^']*\)'.*/\1/p" "$$log" | head -1); \
		if [ -n "$$miss" ] && grep -qF "$$(basename $$miss)" .gitignore; then skipped="$$skipped $$1"; \
		else fail="$$fail $$1"; fi; \
	}; \
	for w in $(WORLDS); do run "$$w" ""; done; \
	echo "-- ros2_ws worlds $$note"; \
	for w in $(ROS_WORLDS); do run "$$w" "$$mute"; done; \
	rm -f "$$log"; \
	if [ -n "$$skipped" ]; then echo; \
		echo "SKIPPED -- needs an asset that is generated, not committed (external/external_assets.yaml):"; \
		for w in $$skipped; do echo "  $$w"; done; fi; \
	if [ -n "$$fail" ]; then echo; echo "FAILED ($$n worlds run):"; for w in $$fail; do echo "  $$w"; done; exit 1; \
	else echo; echo "$$n worlds run headless for $(SMOKE_STEPS) steps; none failed."; fi

.PHONY: check
check:  ## Publication hygiene: no LFS, every asset folder attributed, no absolute paths
	@$(PY) tools/check_repo.py

.PHONY: ci
ci: lint test smoke check doc  ## Everything CI runs, in CI's order

.PHONY: doc
doc:  ## Build the Sphinx HTML docs into build/html
	$(SPHINX) -b html -W docs build/html
	@echo "docs -> build/html/index.html"

.PHONY: thumbnails
thumbnails:  ## Render each model's preview as <model-dir>/<name>.thumb.png (run when models change)
	$(PY) roqsim_assets/tools/render_thumbnails.py
	@echo "thumbnails written beside each model (commit them; not a dependency of 'make doc')"

.PHONY: view-doc
view-doc: doc  ## Build the docs and open them in the default browser
	@html="file://$(abspath build/html/index.html)"; \
	echo "opening $$html"; \
	if command -v xdg-open >/dev/null 2>&1; then xdg-open "$$html" >/dev/null 2>&1 & \
	elif command -v open >/dev/null 2>&1; then open "$$html"; \
	else echo "open $$html manually (no xdg-open/open found)"; fi

.PHONY: format
format:  ## Format + autofix all Python with ruff
	$(RUFF) format $(SRC)
	$(RUFF) check --fix $(SRC)

.PHONY: lint
lint:  ## Check formatting/lint with ruff (no changes)
	$(RUFF) format --check $(SRC)
	$(RUFF) check $(SRC)

.PHONY: external-list
external-list:  ## List external assets (sources + generated targets) from external_assets.yaml
	$(EXTERNAL) list

.PHONY: external-fetch
external-fetch:  ## Fetch external-asset sources (manual sources must be placed by hand)
	$(EXTERNAL) fetch $(if $(RESOURCE),--resource $(RESOURCE))

.PHONY: external-resources
external-resources:  ## Fetch + convert external assets into their (git-ignored) targets [RESOURCE=name]
	$(EXTERNAL) convert $(if $(RESOURCE),--resource $(RESOURCE))

.PHONY: external-sync-gitignore
external-sync-gitignore:  ## Rewrite the managed .gitignore block from external_assets.yaml
	$(EXTERNAL) sync-gitignore

.PHONY: add-external-resource
add-external-resource:  ## Append a resource to external_assets.yaml + .gitignore (see `$(EXTERNAL) add -h`)
	$(EXTERNAL) add $(ARGS)

.PHONY: clean
clean:  ## Remove venv, docs build, colcon artifacts, and caches
	rm -rf $(VENV) build
	rm -rf ros2_ws/build ros2_ws/install ros2_ws/log
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} + 2>/dev/null || true
