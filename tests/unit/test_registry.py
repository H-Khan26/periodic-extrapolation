from __future__ import annotations

import pytest

from periodic.registry import Registry, RegistryError


def test_register_get_and_names() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("b")(2)
    reg.register("a")(1)
    assert reg.get("a") == 1
    assert reg.names() == ["a", "b"]
    assert "a" in reg
    assert len(reg) == 2
    assert list(reg) == ["a", "b"]


def test_duplicate_and_unknown_names_raise_with_context() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("a")(1)
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("a")(2)
    with pytest.raises(RegistryError, match="unknown thing 'zzz'; known: a"):
        reg.get("zzz")
