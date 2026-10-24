"""Minimal name -> object registry used for targets and models."""

from __future__ import annotations

from collections.abc import Callable


class RegistryError(KeyError):
    """Raised for unknown or duplicate registry names."""

    def __str__(self) -> str:  # KeyError quotes its message by default
        return str(self.args[0]) if self.args else ""


class Registry[T]:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise RegistryError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise RegistryError(f"unknown {self.kind} '{name}'; known: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._items)
