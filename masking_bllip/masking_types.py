# Copyright 2021-2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Types used by the masking rules."""

import collections

Chunk = collections.namedtuple(
    "Chunk",
    [
        "seq_idx",
        "inputs",
        "inputs_ttypes",
        "labels",
        "labels_ttypes",
        "attn_mask",
        "attn_relpos",
        "attn_indicator",
        "memory_attn_mask",
        "memory_padding_mask",
        "memory_pos",
        "depth",
        "composed_position",
        "beginning_of_seq",
        "end_of_seq",
        "smartmem_mem_from_seq",
        "smartmem_mem_from_mem",
    ],
)
