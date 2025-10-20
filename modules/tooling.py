# modules/tooling.py
from typing import List, Dict, Any
from . import df_tools, kb

TOOLS: List[Dict[str, Any]] = []
TOOLS.extend(getattr(df_tools, "TOOLS", []))
TOOLS.extend(getattr(kb, "KB_TOOLS", []))

REGISTRY: Dict[str, Any] = {}
REGISTRY.update(getattr(df_tools, "REGISTRY", {}))
REGISTRY.update(getattr(kb, "KB_REGISTRY", {}))

def tool_names():
    return [t["function"]["name"] for t in TOOLS]
