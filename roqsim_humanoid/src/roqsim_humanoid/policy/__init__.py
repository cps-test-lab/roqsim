"""Bundled locomotion policies + their deploy configs.

**G1** -- ``motion.pt`` is the pretrained TorchScript G1 walking policy and ``g1.yaml`` its deploy
config (PD gains, default joint angles, observation/action scales, timing) -- both vendored verbatim
from unitree_rl_gym (BSD-3-Clause). The ``g1_locomotion`` plugin loads them from :data:`POLICY_DIR`.

**Oli** -- ``oli/policy.onnx`` is the pretrained ONNX whole-body walk policy for the LimX Oli
(HU_D04_01) and ``oli/walk_param.yaml`` its deploy config, vendored verbatim from
humanoid-rl-deploy-python (Apache-2.0). The ``oli_locomotion`` plugin loads them from
:data:`OLI_POLICY` / :data:`OLI_CONFIG`.

Both run on exactly the conventions their policies were trained on; see the family THIRD_PARTY.md.
"""

from __future__ import annotations

from pathlib import Path

POLICY_DIR = Path(__file__).parent
DEFAULT_POLICY = POLICY_DIR / "motion.pt"
DEFAULT_CONFIG = POLICY_DIR / "g1.yaml"

OLI_POLICY = POLICY_DIR / "oli" / "policy.onnx"
OLI_CONFIG = POLICY_DIR / "oli" / "walk_param.yaml"


def find_spec(name: str) -> Path:
    """Resolve a ``policy:`` name to its spec file, e.g. ``g1_stand`` -> ``g1_stand/g1_stand.spec.yaml``.

    Deliberately a naming convention rather than a registry: adding a policy is dropping a directory
    here, with no code to edit and nothing to register. An absolute or relative path is passed through,
    so an out-of-tree policy needs no blessing either.
    """
    path = Path(name)
    if path.suffix in (".yaml", ".yml"):
        return path if path.is_absolute() else (POLICY_DIR / path)
    return POLICY_DIR / name / f"{name}.spec.yaml"
