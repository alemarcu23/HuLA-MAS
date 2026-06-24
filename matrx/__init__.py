# Aliasing imports
from matrx.world_builder import WorldBuilder


__docformat__ = "restructuredtext"


######
# We do this so we are sure everything is imported and thus can be found
# noinspection PyUnresolvedReferences
import pkgutil
import importlib

__all__ = []
_skipped_modules = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    short_name = module_name.split(".")[-1]
    try:
        _module = importlib.import_module(module_name)
    except Exception as _exc:
        # Optional/extension modules (e.g. the LLM agents under matrx.agents.llm)
        # may have unmet dependencies. Skip them here so that importing `matrx`
        # and running the rule-based simulation still works; they can still be
        # imported explicitly by code that actually needs them.
        _skipped_modules.append((module_name, repr(_exc)))
        continue
    __all__.append(short_name)
    globals()[short_name] = _module

if _skipped_modules:
    import warnings as _warnings
    _warnings.warn(
        "matrx: skipped auto-import of {} optional module(s) with unmet "
        "dependencies: {}".format(
            len(_skipped_modules),
            ", ".join(name for name, _ in _skipped_modules),
        ),
        stacklevel=2,
    )
######

# Set package attributes
name = "MATRX: Man-Agent Teaming - Rapid Experimentation Software"
