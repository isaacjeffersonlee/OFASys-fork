# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from ofasys.configure import auto_import

from .base import BaseMetric

__all__ = [
    'BaseMetric',
]
auto_import(__file__)
