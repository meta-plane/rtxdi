/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RTXDI_RESERVOIR_ADDRESSING_HLSLI
#define RTXDI_RESERVOIR_ADDRESSING_HLSLI

uint2 RTXDI_PixelPosToReservoirPos(uint2 pixelPosition, uint activeCheckerboardField)
{
    if (activeCheckerboardField == 0)
        return pixelPosition;

    return uint2(pixelPosition.x >> 1, pixelPosition.y);
}

uint2 RTXDI_ReservoirPosToPixelPos(uint2 reservoirIndex, uint activeCheckerboardField)
{
    if (activeCheckerboardField == 0)
        return reservoirIndex;

    uint2 pixelPosition = uint2(reservoirIndex.x << 1, reservoirIndex.y);
    pixelPosition.x += ((pixelPosition.y + activeCheckerboardField) & 1);
    return pixelPosition;
}

uint RTXDI_ReservoirPositionToPointer(
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint2 blockIdx = reservoirPosition / RTXDI_RESERVOIR_BLOCK_SIZE;
    uint2 positionInBlock = reservoirPosition % RTXDI_RESERVOIR_BLOCK_SIZE;

    return reservoirArrayIndex * reservoirParams.reservoirArrayPitch
        + blockIdx.y * reservoirParams.reservoirBlockRowPitch
        + blockIdx.x * (RTXDI_RESERVOIR_BLOCK_SIZE * RTXDI_RESERVOIR_BLOCK_SIZE)
        + positionInBlock.y * RTXDI_RESERVOIR_BLOCK_SIZE
        + positionInBlock.x;
}

// phgphg: comment 보강
// Permutes the previous-frame pixel position so that adjacent pixels reference
// different temporal neighbors, breaking correlation for spatial resampling.
//
// The net displacement per axis is ((p + offset) ^ 3) - (p + offset), where
// p is the original coordinate and offset is derived from uniformRandomNumber.
// Because XOR 3 only flips the lowest two bits, the displacement depends on
// (p + offset) & 3 and is always one of {-3, -1, +1, +3}:
//
//   (p + offset) & 3 | displacement
//   -----------------+-------------
//          0         |     +3
//          1         |     +1
//          2         |     -1
//          3         |     -3
//
// Key properties:
//   - Zero displacement never occurs, so a pixel never re-reads its own reservoir.
//   - Only odd displacements occur, which naturally swaps the checkerboard field
//     (even/odd parity flips).
//   - Different pixels get different displacements because p varies per pixel,
//     even though offset is uniform across all pixels in the same frame.
//     This is why the XOR cannot be factored out of the (p + offset) expression:
//     XOR does not distribute over addition.
//   - Changing uniformRandomNumber each frame rotates which displacement each
//     pixel receives, decorrelating the permutation over time.
void RTXDI_ApplyPermutationSampling(inout int2 prevPixelPos, uint uniformRandomNumber)
{
    int2 offset = int2(uniformRandomNumber & 3, (uniformRandomNumber >> 2) & 3);
    prevPixelPos += offset;

    prevPixelPos.x ^= 3;
    prevPixelPos.y ^= 3;

    prevPixelPos -= offset;
}

#endif // RTXDI_RESERVOIR_ADDRESSING_HLSLI
