#!/bin/bash -e
# Build the roqsim container images: a thin wrapper around `docker build` with --project
# (registry prefix), --push and --ros-distro. Self-contained -- the build context is this repo's
# root, so it honours ./.dockerignore and reaches into nothing outside this repository.
#
# Note it is a plain `docker build`, single-architecture, for the host it runs on. A consumer
# needing multi-arch images drives buildx itself; nothing here reads an architecture policy.
#
# Usage:
#   ./container/build.sh [--image roqsim|roqsim-ros|all] [--project <prefix>] \
#                        [--ros-distro <distro>] [--push] [-- <extra docker build args>]
#
#   --image        Which image(s) to build. Default: all.
#   --project      Registry/namespace prefix for the tag+push (e.g. ghcr.io/cps-test-lab/). Optional.
#   --ros-distro   ROS distro for roqsim-ros. Default: jazzy.
#   --push         docker push after building.

BASEDIR=$(cd "$(dirname "$0")" && pwd)
CONTEXT=$(cd "${BASEDIR}/.." && pwd)   # roqsim repo root

ROS_DISTRO="jazzy"
PROJECT=""
IMAGE="all"
PUSH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --image)
      IMAGE="$2"; shift 2 ;;
    --project)
      PROJECT="$2"; shift 2 ;;
    --ros-distro)
      ROS_DISTRO="$2"; shift 2 ;;
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

echo "Context:    ${CONTEXT}"
echo "Image(s):   ${IMAGE}"
echo "ROS distro: ${ROS_DISTRO}"
echo "Project:    ${PROJECT:-<none>}"

build_image() {
  local name="$1" dockerfile="$2"; shift 2
  local tag="${name}:latest"
  echo "==> Building ${tag} (-f ${dockerfile})"
  DOCKER_BUILDKIT=1 docker build \
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
