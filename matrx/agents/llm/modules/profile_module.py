from typing import Any, Dict, List, Optional

from matrx.agents.capabilities.capability import (
    VisionCapability,
    StrengthCapability,
    MedicalCapability,
    SpeedCapability,
)

MSG_CAPABILITY_CLAIM = 'capability_claim'
ROLE_CLAIM_MSG_TYPE = 'role_claim'
KNOWLEDGE_INFORMED = 'informed'

DISCOVERY_NOTE = (
    "You do not yet know your exact capability limits. "
    "Attempt tasks; if you fail, the critic will tell you why and you should adjust accordingly."
)

_CAPABILITY_CLASSES = {
    'vision': VisionCapability,
    'strength': StrengthCapability,
    'medical': MedicalCapability,
    'speed': SpeedCapability,
}

CAPABILITIES_MAP = {dim: set(cls.levels) for dim, cls in _CAPABILITY_CLASSES.items()}


def resolve_capabilities(preset_or_dict, presets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(preset_or_dict, str):
        if preset_or_dict not in presets:
            raise ValueError(
                f"Unknown preset '{preset_or_dict}'. Available: {list(presets.keys())}"
            )
        caps = dict(presets[preset_or_dict])
    elif isinstance(preset_or_dict, dict):
        caps = dict(preset_or_dict)
    else:
        raise ValueError(f"Capabilities must be a preset name or a dict, got {type(preset_or_dict)}")

    for dim, valid_vals in CAPABILITIES_MAP.items():
        if caps.get(dim) not in valid_vals:
            raise ValueError(f"Invalid capability '{dim}': {caps.get(dim)}. Valid: {valid_vals}")
    return caps


def get_capability_summary(capabilities: Optional[Dict[str, Any]]) -> str:
    if not capabilities:
        return 'unknown'
    v = capabilities.get('vision', '?')
    s = capabilities.get('strength', '?')
    m = capabilities.get('medical', '?')
    sp = capabilities.get('speed', '?')
    return f"vision:{v} strength:{s} medical:{m} speed:{sp}"


class Profile:

    def __init__(
        self,
        capabilities: Dict[str, Any],
        roles: Optional[List[str]] = None,
        knowledge: str = KNOWLEDGE_INFORMED,
        goal: str = '',
    ) -> None:
        self._caps = dict(capabilities)
        self.vision = VisionCapability(self._caps['vision'])
        self.strength = StrengthCapability(self._caps['strength'])
        self.medical = MedicalCapability(self._caps['medical'])
        self.speed = SpeedCapability(self._caps['speed'])
        self.roles: List[str] = list(roles) if roles else []
        self.knowledge = knowledge
        self.goal = goal

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._caps)

    def capability_prompt(self) -> str:
        if self.knowledge != KNOWLEDGE_INFORMED:
            return DISCOVERY_NOTE
        lines = ["Your agent capabilities:"]
        lines += self.vision.describe()
        lines += self.medical.describe()
        lines += self.strength.describe()
        lines += self.speed.describe()
        return '\n'.join(lines)

    def role_goal(self) -> str:
        return self.goal

    def role_str(self) -> str:
        return ', '.join(self.roles) if self.roles else 'unassigned'
