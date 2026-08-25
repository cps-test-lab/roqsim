#!/bin/bash -e
# Build the roqsim container images: a thin wrapper around `docker build` with --project
# (registry prefix), --push and --ros-distro. Self-contained -- the build context is this repo's
# root, so it honours ./.dockerignore and reaches into nothing outside this repository.
#
# By default this builds for the host it runs on, which is what a local test wants. --multiarch
# instead builds every architecture container/platforms.env lists for the image, and that file is
# the same one .github/workflows/image.yml reads: encoding the architecture policy here as well is
# how a local build and a CI build come to disagree about what an image is.
#
# Note that --multiarch requires --push. A multi-platform build produces an index, and the local
# daemon can hold only one image, so buildx has nothing to --load; without a registry to write to
# there is nowhere for the result to go. Refusing is better than silently building one arch.
#
# Usage:
#   ./container/build.sh [--image roqsim|roqsim-ros|all] [--project <prefix>] \
#                        [--ros-distro <distro>] [--platform <list>] [--multiarch] [--push] \
#                        [-- <extra docker build args>]
#
#   --image        Which image(s) to build. Default: all.
#   --project      Registry/namespace prefix for the tag+push (e.g. ghcr.io/cps-test-lab/). Optional.
#   --ros-distro   ROS distro for roqsim-ros. Default: jazzy.
#   --platform     Explicit platform list (e.g. linux/arm64). Overrides --multiarch.
#   --multiarch    Build every platform container/platforms.env lists for the image. Needs --push.
#   --push         docker push after building.

BASEDIR=$(cd "$(dirname "$0")" && pwd)
CONTEXT=$(cd "${BASEDIR}/.." && pwd)   # roqsim repo root

ROS_DISTRO="jazzy"
PROJECT=""
IMAGE="all"
PUSH=""
PLATFORM=""
MULTIARCH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --image)
      IMAGE="$2"; shift 2 ;;
    --project)
      PROJECT="$2"; shift 2 ;;
    --ros-distro)
      ROS_DISTRO="$2"; shift 2 ;;
    --platform)
      PLATFORM="$2"; shift 2 ;;
    --multiarch)
      MULTIARCH=1; shift ;;
    --push|-n)
      PUSH=1; shift ;;
    --)
      shift; break ;;
    *)
      break ;;
  esac
done

# Remaining args are passed straight through to docker build.
EXTRA_ARGS="$@"

# Ensure PROJECT ends with a slash when non-empty.
if [[ -n "${PROJECT}" ]]; then
  [[ "${PROJECT}" == */ ]] || PROJECT="${PROJECT}/"
fi

if [[ -n "${MULTIARCH}" && -z "${PUSH}" ]]; then
  echo "ERROR: --multiarch needs --push: a multi-platform build cannot be loaded into the local" >&2
  echo "       daemon, so without a registry there is nowhere to put the result." >&2
  exit 1
fi
if [[ -n "${MULTIARCH}" && -z "${PROJECT}" ]]; then
  echo "ERROR: --multiarch needs --project: pushing an index requires a registry to push it to." >&2
  exit 1
fi

# The architecture policy, shared with .github/workflows/image.yml. Sourced rather than restated.
# shellcheck source=platforms.env
source "${BASEDIR}/platforms.env"

# The platform list for one image: an explicit --platform wins, then --multiarch consults the
# policy, and otherwise we say nothing and docker builds for this host.
platforms_for() {
  local name="$1"
  if [[ -n "${PLATFORM}" ]]; then echo "${PLATFORM}"; return; fi
  if [[ -z "${MULTIARCH}" ]]; then echo ""; return; fi
  case "${name}" in
    roqsim)     echo "${PLATFORMS_ROQSIM}" ;;
    roqsim-ros) echo "${PLATFORMS_ROQSIM_ROS}" ;;
    *)
      echo "ERROR: no platform policy in container/platforms.env for image '${name}'" >&2
      exit 1 ;;
  esac
}

echo "Context:    ${CONTEXT}"
echo "Image(s):   ${IMAGE}"
echo "ROS distro: ${ROS_DISTRO}"
echo "Project:    ${PROJECT:-<none>}"

build_image() {
  local name="$1" dockerfile="$2"; shift 2
  local tag="${name}:latest"
  local platforms; platforms=$(platforms_for "${name}")

  # A multi-platform build has to go straight to the registry under its final name: there is no
  # local image to `docker tag` afterwards. A host build keeps the old two-step behaviour so an
  # untagged local build still works with no --project.
  if [[ -n "${platforms}" && "${platforms}" == *,* ]]; then
    echo "==> Building ${PROJECT}${tag} for ${platforms} (-f ${dockerfile})"
    DOCKER_BUILDKIT=1 docker buildx build \
      --platform "${platforms}" \
      "$@" \
      ${EXTRA_ARGS} \
      -t "${PROJECT}${tag}" \
      -f "${dockerfile}" \
      --push \
      "${CONTEXT}"
    return
  fi

  echo "==> Building ${tag}${platforms:+ for ${platforms}} (-f ${dockerfile})"
  DOCKER_BUILDKIT=1 docker build \
    ${platforms:+--platform "${platforms}"} \
    "$@" \
    ${EXTRA_ARGS} \
    -t "${tag}" \
    -f "${dockerfile}" \
    "${CONTEXT}"

  if [[ -n "${PROJECT}" ]]; then
    docker tag "${tag}" "${PROJECT}${tag}"
  fi

  if [[ -n "${PUSH}" ]]; then
    echo "==> Pushing ${PROJECT}${tag}"
    docker push "${PROJECT}${tag}"
  fi
}

if [[ "${IMAGE}" == "roqsim" || "${IMAGE}" == "all" ]]; then
  build_image "roqsim" "${BASEDIR}/Dockerfile"
fi

if [[ "${IMAGE}" == "roqsim-ros" || "${IMAGE}" == "all" ]]; then
  build_image "roqsim-ros" "${BASEDIR}/Dockerfile.ros" --build-arg "ROS_DISTRO=${ROS_DISTRO}"
fi
