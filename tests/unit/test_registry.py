from __future__ import annotations

import pytest

from periodic.registry import Registry, RegistryError


def test_register_get_and_reject_duplicate_or_unknown_names() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("b")(2)
    reg.register("a")(1)
    assert reg.get("a") == 1
    assert reg.names() == ["a", "b"]
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("a")(2)
    with pytest.raises(RegistryError, match="unknown thing 'zzz'; known: a, b"):
        reg.get("zzz")
