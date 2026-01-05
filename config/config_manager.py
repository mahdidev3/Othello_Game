from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ConfigManager:
    """
    Lightweight config wrapper supporting dot-path access:

        cfg.get("mcts.iterations")
        cfg.get("training.learning_rate", default=1e-3)

    Accepts a nested dict-like config. This class is intentionally tiny and fast.
    """
    data: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        if not key:
            return self.data
        cur: Any = self.data
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        cur = self.data
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = value
