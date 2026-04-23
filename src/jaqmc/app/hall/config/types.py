# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from .disk import HallDiskConfig
from .spherical import HallSphericalConfig
from .torus import HallTorusConfig

type HallSystemConfig = HallDiskConfig | HallSphericalConfig | HallTorusConfig
