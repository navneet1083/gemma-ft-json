"""Tiny plugin registry -> the 'plug-and-play' mechanism of the codebase.

Register alternative vision encoders / projectors / losses by name and pick
them from YAML, so fusing a different model later means writing ONE new
class + ONE config line, with zero edits to the trainer.

Example
-------
    VISION_ENCODERS = Registry("vision_encoder")

    @VISION_ENCODERS.register("table_vit_lite")
    class TableViTLite(nn.Module): ...

    enc_cls = VISION_ENCODERS.get(cfg_name)
"""
from typing import Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._store: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        def deco(obj: Callable) -> Callable:
            if name in self._store:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._store[name] = obj
            return obj
        return deco

    def get(self, name: str) -> Callable:
        if name not in self._store:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._store)}")
        return self._store[name]

    def names(self):
        return sorted(self._store)
