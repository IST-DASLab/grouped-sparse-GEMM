/***************************************************************************************************
 * Copyright (C) 2026 Kwanhee Lee and Dan Alistarh. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 **************************************************************************************************/

// Selects the arch configuration for this translation unit. setup.py compiles one extension module
// per target arch and passes -DPAIRED_NVFP4_SM=<sm> to every TU of that module.

#pragma once

#ifndef PAIRED_NVFP4_SM
#error "PAIRED_NVFP4_SM must be defined (setup.py passes it per extension module)"
#endif

#if PAIRED_NVFP4_SM == 100
#include "sm100/config.cuh"
#elif PAIRED_NVFP4_SM == 120
#include "sm120/config.cuh"
#else
#error "no arch configuration for this PAIRED_NVFP4_SM (see docs/porting.md)"
#endif
