# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os as _os

_env_root = _os.environ.get("OPENARM_ROOT_DIR")
if _env_root:
    OPENARM_ROOT_DIR = _os.path.abspath(_os.path.expanduser(_env_root))
else:
    OPENARM_ROOT_DIR = _os.path.dirname(_os.path.abspath(__file__))

__all__ = ["OPENARM_ROOT_DIR"]

# Register Gym environments.
try:
    from .tasks import *
except ModuleNotFoundError:
    pass

# Register UI extensions.
try:
    from .ui_extension_example import *
except ModuleNotFoundError:
    pass
