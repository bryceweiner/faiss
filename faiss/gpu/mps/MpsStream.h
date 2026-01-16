/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Metal/Metal.h>

namespace faiss {
namespace gpu {

struct MpsStream {
    id<MTLCommandQueue> queue;
};

} // namespace gpu
} // namespace faiss
