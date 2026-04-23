# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jaqmc.utils.config import configurable_dataclass

__all__ = ["HallTorusConfig"]


@configurable_dataclass
class HallTorusConfig:
    flux: int = 2
    nspins: tuple[int, int] = (3, 0)
    interaction_strength: float = 1.0
