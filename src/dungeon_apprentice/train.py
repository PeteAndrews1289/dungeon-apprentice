"""Backwards-compatible shim for the former single-file ``train`` module.

The implementation now lives in :mod:`dungeon_apprentice.training`. This module
re-exports it so existing imports and the ``dungeon-train`` console script keep
working unchanged.
"""

from __future__ import annotations

from dungeon_apprentice.training import *  # noqa: F403
from dungeon_apprentice.training import __all__ as _training_all

__all__ = list(_training_all)
