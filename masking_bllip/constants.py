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

"""Constants for masking code."""

# constexpr int32_t kPad = 0;
# constexpr int32_t kSos = 1;
# constexpr int32_t kRightArc = 15;
# constexpr int32_t kLeftArc = 4;
# constexpr int32_t kRightArc2 = 16;
# constexpr int32_t kLeftArc2 = 14;
# constexpr int32_t kPopRoot = 2;
# constexpr int32_t kOpeningNonTerminal = 2;
# constexpr int32_t kClosingNonTerminal = 4;
# constexpr int32_t kClosingNonTerminal2 = 14;

import enum

PAD = 0
SOS = 1
# OPENING_NT = 2
POPROOT = 2
TERMINAL = 3
# CLOSING_NT = 4
LEFTARC = 4
PLACEHOLDER = 5
STARTOFWORD = 6
LEFTARC_2 = 14
RIGHTARC = 15
RIGHTARC_2 = 16

TOKEN_TYPES = [PAD, SOS, POPROOT, TERMINAL, STARTOFWORD, LEFTARC, RIGHTARC]


class TokenTypesEnum(enum.IntEnum):
  PAD = PAD
  SOS = SOS
  POPROOT = POPROOT
  TERMINAL = TERMINAL
  STARTOFWORD = STARTOFWORD
  LEFTARC = LEFTARC
  RIGHTARC = RIGHTARC
  PLACEHOLDER = PLACEHOLDER


# For proposals involving duplicating tokens.
# CLOSING_NT_2 = 14
