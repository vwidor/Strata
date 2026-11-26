#include "strata/core/bw_dense_defaults.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/window_limits.hpp"
#include "s26_tsum.cuh"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

// The FP32-activation MMVF implementation below is adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/{mmvf.cu,common.cuh}.
// Scope: ordinary contiguous BF16 matrix, one FP32 activation vector, no fusion/ids/channels.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
__device__ __forceinline__ float mmvf_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

// STRATA_MMVF_CHECK=1 (diagnostic): the constant-trip fast path also runs the plain loop with volatile loads (so the loads happen again, later) and
// compares each thread's accumulator bit for bit: g_mmvf_chk[0] checks, [1] differing accumulators, [2..4] block / thread / n_in of the first one.
// A difference means the activations or weights changed between the early and the late loads (no arithmetic differs between the two loops).
__device__ unsigned long long g_mmvf_chk[8];
__device__ __forceinline__ void mmvf_chk_note(float a, float b, int blk, int t, int n_in) {
    atomicAdd(&g_mmvf_chk[0], 1ull);
    if (__float_as_uint(a) != __float_as_uint(b)) {
        if (atomicAdd(&g_mmvf_chk[1], 1ull) == 0ull) { g_mmvf_chk[2] = (unsigned long long) blk; g_mmvf_chk[3] = (unsigned long long) t; g_mmvf_chk[4] = (unsigned long long) n_in;
                                                       g_mmvf_chk[5] = __float_as_uint(a); g_mmvf_chk[6] = __float_as_uint(b); }
    }
}

template <int BLOCK_SIZE>
__global__ void bf16_f32_mmvf_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                    float* __restrict__ y, int n_in, int fast) {
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const float2* inputs2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if (t < 32) partials[t] = 0.0f;
    __syncthreads();
    float acc = 0.0f;
    if (BLOCK_SIZE == 256 && fast != 0 && n_in == 2560) {
        // STRATA_SMALL_UNR bit 0: the router-shaped single-column read, five passes of one weight pair each (all in range), constant
        // trip count so the weight and activation loads of the five passes are in flight together.  The same FMAs in the same order.
        uint32_t wv[5];
        float2 xv[5];
#pragma unroll
        for (int qq = 0; qq < 5; ++qq) { wv[qq] = weights2[t + qq * BLOCK_SIZE]; xv[qq] = inputs2[t + qq * BLOCK_SIZE]; }
#pragma unroll
        for (int qq = 0; qq < 5; ++qq) {
            acc = __fmaf_rn(f32_from_bf16((uint16_t) wv[qq]), xv[qq].x, acc);
            acc = __fmaf_rn(f32_from_bf16((uint16_t) (wv[qq] >> 16)), xv[qq].y, acc);
        }
        if (fast == 2) {
            float acc2 = 0.0f;
            for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
                const uint32_t weight = *(const volatile uint32_t*) (weights2 + pair);
                const volatile float* ip = reinterpret_cast<const volatile float*>(inputs2 + pair);
                const float2 input = make_float2(ip[0], ip[1]);
                acc2 = __fmaf_rn(f32_from_bf16((uint16_t) weight), input.x, acc2);
                acc2 = __fmaf_rn(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc2);
            }
            mmvf_chk_note(acc, acc2, (int) blockIdx.x, t, n_in);
        }
    } else
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float2 input = inputs2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        acc = __fmaf_rn(f32_from_bf16((uint16_t) weight), input.x, acc);
        acc = __fmaf_rn(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc);
    }
    acc = mmvf_warp_sum(acc);
    if constexpr (BLOCK_SIZE > 32) {
        // All lanes have the same reduced value; one store avoids a same-value shared-memory race.
        if ((t & 31) == 0) partials[t / 32] = acc;
        __syncthreads();
        if (t < 32) acc = mmvf_warp_sum(partials[t]);
    }
    if (t == 0) y[blockIdx.x] = acc;
}

// the same kernel for up to 8 activation rows - the weight row is read ONCE and every
// token keeps its own accumulator with exactly the single-row kernel's order (pairs, two ordered FMAs, the same warp
// and block reductions), so each output is bit-identical to a bf16_f32_mmvf_kernel launch of its own.
template <int BLOCK_SIZE, int NT, bool EXACT_T = true>
__global__ void bf16_f32_mmvf_multi_kernel(const float* x_, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* y_, int64_t ldy, int n_in, int n_tok, int fast) {
    const float* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    float* STRATA_PDL_RESTRICT y = y_;
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    __shared__ float partials[NT][32];
    if (t < 32) {
#pragma unroll
        for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
    }
    __syncthreads();
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    // PDL (pdl.hpp): this thread's first PRE weight pairs are loaded before waiting for the activations
    constexpr int PRE = kPdlPrefetch ? 8 : 0;
    uint32_t wpre[PRE > 0 ? PRE : 1];
#pragma unroll
    for (int q = 0; q < PRE; ++q) {
        const int pair = t + q * BLOCK_SIZE;
        wpre[q] = pair < n_in / 2 ? __ldg(weights2 + pair) : 0u;
    }
    pdl_wait();
    if (BLOCK_SIZE == 256 && fast != 0 && n_in == 2560) {
        // BW-DENSE (STRATA_SMALL_UNR bit 0): the router-shaped read (2560 inputs, 256 threads: five passes of one weight pair each, all
        // in range) with constant trip counts, so the activation loads of every pass are in flight together instead of one memory
        // latency a pass.  The same FMAs in the same order (pass ascending, w0 then w1): bitwise the loop below.
        float2 xin[NT][5];
#pragma unroll
        for (int qq = 0; qq < 5; ++qq)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (EXACT_T || k < n_tok) xin[k][qq] = __ldg(reinterpret_cast<const float2*>(x + (size_t) k * ldx) + (t + qq * BLOCK_SIZE));
#pragma unroll
        for (int qq = 0; qq < 5; ++qq) {
            uint32_t weight;
            if constexpr (PRE >= 5) weight = wpre[qq];
            else weight = __ldg(weights2 + t + qq * BLOCK_SIZE);
            const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (EXACT_T || k < n_tok) {
                    acc[k] = __fmaf_rn(w0, xin[k][qq].x, acc[k]);
                    acc[k] = __fmaf_rn(w1, xin[k][qq].y, acc[k]);
                }
            }
        }
        if (fast == 2) {
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (EXACT_T || k < n_tok) {
                    float acc2 = 0.0f;
                    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
                        const uint32_t weight = *(const volatile uint32_t*) (weights2 + pair);
                        const volatile float* ip = reinterpret_cast<const volatile float*>(reinterpret_cast<const float2*>(x + (size_t) k * ldx) + pair);
                        const float2 input = make_float2(ip[0], ip[1]);
                        acc2 = __fmaf_rn(f32_from_bf16((uint16_t) weight), input.x, acc2);
                        acc2 = __fmaf_rn(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc2);
                    }
                    mmvf_chk_note(acc[k], acc2, (int) blockIdx.x, t, n_in);
                }
            }
        }
    } else {
    int q = 0;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE, ++q) {
        const uint32_t weight = q < PRE ? wpre[q] : __ldg(weights2 + pair);
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (EXACT_T || k < n_tok) {
                const float2 input = __ldg(reinterpret_cast<const float2*>(x + (size_t) k * ldx) + pair);
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
    }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k)
        if (EXACT_T || k < n_tok) acc[k] = mmvf_warp_sum(acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (EXACT_T || k < n_tok) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (EXACT_T || k < n_tok) acc[k] = mmvf_warp_sum(partials[k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (EXACT_T || k < n_tok) y[(size_t) k * ldy + blockIdx.x] = acc[k];
}

// S25 (STRATA_MMVF_ROWS=1): RPB output rows per block. Each block read its row's weights once but all T activation
// rows again (n_out blocks x T x n_in floats through L2 - the part that grew with T); here a block reads the
// activations once for RPB rows. Per output, thread t still walks pairs t, t + BLOCK_SIZE, ... with the same two
// ordered FMAs and the same warp and block reductions: bit-identical to bf16_f32_mmvf_multi_kernel.
// S26 STRATA_LFUSE=1 (AUX): one more block computes row 0 of w_aux into y_aux[k * ldy_aux] - the same per-output
// code, so each aux output is bitwise what its own bf16_f32_mmvf_multi_kernel launch gave (the shared expert's gate)
template <int BLOCK_SIZE, int NT, int RPB, bool TS = false, bool AUX = false>
__global__ void bf16_f32_mmvf_rows_kernel(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* __restrict__ y, int64_t ldy, int n_in, int n_out, int n_tok,
                                          const uint16_t* __restrict__ w_aux = nullptr, float* __restrict__ y_aux = nullptr,
                                          int64_t ldy_aux = 0) {
    const int t = threadIdx.x;
    int o0 = blockIdx.x * RPB;
    if constexpr (AUX) {
        if (blockIdx.x == gridDim.x - 1) { w = w_aux; y = y_aux; ldy = ldy_aux; n_out = 1; o0 = 0; }
    }
    __shared__ float partials[RPB][NT][32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) partials[r][k][t] = 0.0f;
        __syncthreads();
    }
    float acc[RPB][NT];
#pragma unroll
    for (int r = 0; r < RPB; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        float2 in[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) in[k] = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[pair];
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            if (o0 + r >= n_out) break;
            const uint32_t weight = reinterpret_cast<const uint32_t*>(w + (size_t) (o0 + r) * n_in)[pair];
            const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k < n_tok) {
                    acc[r][k] = __fmaf_rn(w0, in[k].x, acc[r][k]);
                    acc[r][k] = __fmaf_rn(w1, in[k].y, acc[r][k]);
                }
            }
        }
    }
    if constexpr (TS) {   // S26 STRATA_TSUM=1: the RPB x NT warp sums (both stages) as transposed butterflies, bitwise the same
        constexpr int V = RPB * NT, P = s26ts::pow2_ceil(V);
        const int lane = t & 31, j = s26ts::tsum_token<P>(lane);
        const bool own = lane == s26ts::tsum_lane<P>(j) && j < V;
        float v[P];
#pragma unroll
        for (int q = 0; q < P; ++q) v[q] = q < V ? acc[q / NT][q % NT] : 0.0f;
        float sum = s26ts::tsum<P>(v, lane);
        if constexpr (BLOCK_SIZE > 32) {
            if (own) partials[j / NT][j % NT][t / 32] = sum;
            __syncthreads();
            if (t >= 32) return;
#pragma unroll
            for (int q = 0; q < P; ++q) v[q] = q < V ? partials[q / NT][q % NT][t] : 0.0f;
            sum = s26ts::tsum<P>(v, lane);
        }
        const int r = j / NT, k = j % NT;
        if (own && o0 + r < n_out && k < n_tok) y[(size_t) k * ldy + o0 + r] = sum;
        return;
    }
#pragma unroll
    for (int r = 0; r < RPB; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = mmvf_warp_sum(acc[r][k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) partials[r][k][t / 32] = acc[r][k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) acc[r][k] = mmvf_warp_sum(partials[r][k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int r = 0; r < RPB; ++r)
            if (o0 + r < n_out)
#pragma unroll
                for (int k = 0; k < NT; ++k)
                    if (k < n_tok) y[(size_t) k * ldy + o0 + r] = acc[r][k];
}

template <int B>
void launch_rows(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int n_in, int n_out, int n_tok,
                 cudaStream_t st, const uint16_t* w_aux = nullptr, float* y_aux = nullptr, int64_t ldy_aux = 0) {
    constexpr int RPB = 4;
    const unsigned nb = (unsigned) ((n_out + RPB - 1) / RPB);
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (w_aux != nullptr) {   // S26 STRATA_LFUSE: + one aux row block
#define S26_AUX(TSV) \
        if (n_tok <= 4) bf16_f32_mmvf_rows_kernel<B, 4, RPB, TSV, true><<<nb + 1, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok, w_aux, y_aux, ldy_aux); \
        else bf16_f32_mmvf_rows_kernel<B, 8, RPB, TSV, true><<<nb + 1, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok, w_aux, y_aux, ldy_aux);
        if (ts) { S26_AUX(true) } else { S26_AUX(false) }
#undef S26_AUX
        return;
    }
    if (ts) {
        if (n_tok <= 4) bf16_f32_mmvf_rows_kernel<B, 4, RPB, true><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
        else bf16_f32_mmvf_rows_kernel<B, 8, RPB, true><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
        return;
    }
    if (n_tok <= 4) bf16_f32_mmvf_rows_kernel<B, 4, RPB><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
    else bf16_f32_mmvf_rows_kernel<B, 8, RPB><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
}

void mmvf_chk_report() {
    unsigned long long h[8] = {};
    if (cudaMemcpyFromSymbol(h, g_mmvf_chk, sizeof(h)) != cudaSuccess) return;
    std::fprintf(stderr, "strata mmvf check: %llu accumulators compared, %llu differ; first: block %llu thread %llu n_in %llu fast %08llx plain %08llx\n", h[0], h[1], h[2], h[3], h[4], h[5], h[6]);
}
// BW-DENSE STRATA_SMALL_UNR bit 0: see bf16_f32_mmvf_multi_kernel (bw_dense_defaults.hpp: on for a 12.0 card, else off; the variable wins)
int mmvf_fast_bit() {
    static int per_dev[64];   // 0 unknown, else bit + 1
    const char* e = std::getenv("STRATA_SMALL_UNR");
    if (e != nullptr) return std::atoi(e) & 1;
#if defined(__HIPCC__)
    return 0;
#else
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (!per_dev[dev]) {
        int ma = 0, mi = 0;
        const bool ok = cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        per_dev[dev] = 1 + (ok ? (strata::small_unr_default(ma, mi) & 1) : 0);
    }
    return per_dev[dev] - 1;
#endif
}
// AMD-Q STRATA_MMVF_CHECK=1 (with the switch on): the kernel also computes the plain read and counts the accumulators that differ (2 = check mode)
int mmvf_fast() {
    const int b = mmvf_fast_bit();
    if (b == 0) return 0;
    static const bool chk = [] {
        const char* c = std::getenv("STRATA_MMVF_CHECK");
        const bool on = c != nullptr && c[0] == '1';
        if (on) std::atexit(mmvf_chk_report);
        return on;
    }();
    return chk ? 2 : b;
}
int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

#if defined(STRATA_HIP_GFX906)
// ---- gfx906 wave64 K-tiled BF16 GEMV (opt-in STRATA_MMVF_WAVE=1/2/4):
// Eliminates the 512-CTA 8-warp LDS reduction & 2 barriers of bf16_f32_mmvf_multi_kernel (which hit 45-48 us on MI50
// at T=1, 2, 8). Four wave64s per block (256 threads), each wave computes R output rows using 128-bit coalesced uint4
// weight loads (8 BF16s unpacked once across all NT tokens) against a 512-float K-tile staged in LDS (2 KiB * NT = 16
// KiB at T=8). Same wave64 reduction across T=1..8 -> every multi-token column is bitwise equal to a single-token call.
template <int NT, int R>
__launch_bounds__(256)
__global__ void bf16_f32_mmvf_wave_kernel(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* __restrict__ y, int64_t ldy, int n_in, int n_out,
                                          const uint16_t* __restrict__ w_aux = nullptr,
                                          float* __restrict__ y_aux = nullptr, int64_t ldy_aux = 0) {
    __shared__ __align__(16) float xs[NT][512];
    const int tid = int(threadIdx.x);
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int row0 = (int(blockIdx.x) * 4 + wave) * R;
    const int total_out = w_aux != nullptr ? (n_out + 1) : n_out;
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = 0.0f;
    for (int k0 = 0; k0 < n_in; k0 += 512) {
        // 256 float2 loads per token -> all 256 threads active even at NT=1, and only 8-byte alignment needed (ldx % 2 == 0).
        for (int idx = tid; idx < NT * 256; idx += 256) {
            const int tk = idx >> 8;
            const int q = idx & 255;
            reinterpret_cast<float2*>(xs[tk])[q] =
                __ldg(reinterpret_cast<const float2*>(x + (size_t) tk * ldx + k0) + q);
        }
        __syncthreads();
        if (row0 < total_out) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int rr = row0 + r;
                if (rr >= total_out) break;
                const uint16_t* wrow = (rr < n_out) ? (w + (size_t) rr * n_in + k0) : (w_aux + k0);
                const uint4 wv = __ldg(reinterpret_cast<const uint4*>(wrow) + lane);
                const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
                const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
                const float w4 = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
                const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    const float4 x0 = reinterpret_cast<const float4*>(xs[k])[lane * 2 + 0];
                    const float4 x1 = reinterpret_cast<const float4*>(xs[k])[lane * 2 + 1];
                    float a = acc[r][k];
                    a = __fmaf_rn(w0, x0.x, a); a = __fmaf_rn(w1, x0.y, a);
                    a = __fmaf_rn(w2, x0.z, a); a = __fmaf_rn(w3, x0.w, a);
                    a = __fmaf_rn(w4, x1.x, a); a = __fmaf_rn(w5, x1.y, a);
                    a = __fmaf_rn(w6, x1.z, a); a = __fmaf_rn(w7, x1.w, a);
                    acc[r][k] = a;
                    if constexpr (R >= 2) __builtin_amdgcn_sched_barrier(0);
                }
            }
        }
        __syncthreads();
    }
    if (row0 >= total_out) return;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = row0 + r;
        if (rr >= total_out) break;
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            float v = acc[r][k];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0) {
                if (rr < n_out) y[(size_t) k * ldy + rr] = v;
                else y_aux[(size_t) k * ldy_aux] = v;
            }
        }
    }
}

int mmvf_wave_mode() {
    static const int m = [] { const char* v = std::getenv("STRATA_MMVF_WAVE"); return v ? std::atoi(v) : 0; }();
    return m;
}

template <int NT>
void launch_mmvf_wave_t(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int n_in, int n_out,
                        int /*r_mode*/, cudaStream_t st, const uint16_t* w_aux = nullptr, float* y_aux = nullptr,
                        int64_t ldy_aux = 0) {
    // R=1 (21-41 VGPRs, 0 spills, 6-10 waves/SIMD) beats R=2/R=4 at every T=1..8 on MI50.
    const size_t total = (size_t) n_out + (w_aux != nullptr ? 1u : 0u);
    const unsigned nb = unsigned((total + 3u) / 4u);
    bf16_f32_mmvf_wave_kernel<NT, 1><<<nb, 256, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, w_aux, y_aux, ldy_aux);
}

bool try_launch_mmvf_wave(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in,
                          int64_t n_out, int n_tok, cudaStream_t st, const uint16_t* w_aux = nullptr,
                          float* y_aux = nullptr, int64_t ldy_aux = 0) {
    const int wm = mmvf_wave_mode();
    if (wm <= 0 || n_out < 512 || n_tok < 1 || n_tok > 8 || (n_in & 511) != 0 || (ldx & 1) != 0 ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 || (reinterpret_cast<uintptr_t>(w) & 15u) != 0 ||
        (w_aux != nullptr && ((reinterpret_cast<uintptr_t>(w_aux) & 15u) != 0 || y_aux == nullptr)))
        return false;
    const int ni = (int) n_in, no = (int) n_out;
    switch (n_tok) {
        case 1: launch_mmvf_wave_t<1>(x, ldx, w, y, ldy, ni, no, wm, st, w_aux, y_aux, ldy_aux); break;
        case 2: launch_mmvf_wave_t<2>(x, ldx, w, y, ldy, ni, no, wm, st, w_aux, y_aux, ldy_aux); break;
        case 3: launch_mmvf_wave_t<3>(x, ldx, w, y, ldy, ni, no, wm, st, w_aux, y_aux, ldy_aux); break;
        case 4: launch_mmvf_wave_t<4>(x, ldx, w, y, ldy, ni, no, wm, st, w_aux, y_aux, ldy_aux); break;
        case 5: launch_mmvf_wave_t<5>(x, ldx, w, y, ldy, ni, no, wm, st, w_aux, y_aux, ldy_aux); break;
        case 6: launch_mmvf_wave_t<6>(x, ldx, w, y, ldy, ni, no, wm, st, w_aux, y_aux, ldy_aux); break;
        case 7: launch_mmvf_wave_t<7>(x, ldx, w, y, ldy, ni, no, wm, st, w_aux, y_aux, ldy_aux); break;
        default: launch_mmvf_wave_t<8>(x, ldx, w, y, ldy, ni, no, wm, st, w_aux, y_aux, ldy_aux); break;
    }
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_wave launch: ") + cudaGetErrorString(result));
    return true;
}
#endif  // STRATA_HIP_GFX906

}  // namespace

bool bf16_gemv_fp32_mmvf_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                   int64_t n_in, int64_t n_out, int n_tok, const uint16_t* w_aux, float* y_aux,
                                   int64_t ldy_aux, void* stream) {
#if defined(STRATA_HIP_GFX906)
    if (x != nullptr && w != nullptr && y != nullptr && w_aux != nullptr && y_aux != nullptr &&
        n_in > 0 && n_out > 0 && n_tok >= 1 && n_tok <= 8 &&
        try_launch_mmvf_wave(x, ldx, w, y, ldy, n_in, n_out, n_tok, (cudaStream_t) stream, w_aux, y_aux, ldy_aux))
        return true;
#endif
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    // exactly the conditions under which both calls would take the rows / multi kernels at the same block size
    if (!rows || n_out < 64 || n_tok < 2 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || w_aux == nullptr || y_aux == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        return false;
    const cudaStream_t st = (cudaStream_t) stream;
    const int ni = (int) n_in, no = (int) n_out;
    switch (mmvf_block_size(n_in)) {
        case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
    }
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi_aux launch: ") + cudaGetErrorString(result));
    return true;
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
#if defined(STRATA_HIP_GFX906)
    if (n_tok >= 1 && n_tok <= 8 && x != nullptr && w != nullptr && y != nullptr && n_in > 0 && n_out > 0 &&
        try_launch_mmvf_wave(x, ldx, w, y, ldy, n_in, n_out, n_tok, (cudaStream_t) stream))
        return;
#endif
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > STRATA_VERIFY_MAX_T || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 ||
        x == nullptr || w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows (16 in a wide build), even n_in/ldx, aligned pointers");
    const cudaStream_t st = (cudaStream_t) stream;
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    if (rows && n_out >= 64 && n_tok <= 8) {   // (the opt-in rows kernels hold 8 columns: a wider window takes the switch below)
        const int ni = (int) n_in, no = (int) n_out;
        switch (mmvf_block_size(n_in)) {
            case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
        }
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
        return;
    }
    // 9..16 rows (wide builds only): one exact instantiation per row count, each row's dot independent of the others.  7 and 8
    // rows stay with the default case below, the instantiation they always had.
#if STRATA_VERIFY_MAX_T > 8
#define STRATA_MMVF_W(N, T) case T: launch_pdl(bf16_f32_mmvf_multi_kernel<N, T, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok, mmvf_fast()); break;
#if STRATA_VERIFY_MAX_T > 12
#define STRATA_MMVF_W13(N) STRATA_MMVF_W(N, 13) STRATA_MMVF_W(N, 14) STRATA_MMVF_W(N, 15) STRATA_MMVF_W(N, 16)
#else
#define STRATA_MMVF_W13(N)
#endif
#define STRATA_MMVF_WIDE(N) \
        STRATA_MMVF_W(N, 9) STRATA_MMVF_W(N, 10) STRATA_MMVF_W(N, 11) STRATA_MMVF_W(N, 12) \
        STRATA_MMVF_W13(N)
#else
#define STRATA_MMVF_WIDE(N)
#endif
#define STRATA_MMVF_M(N) case N: \
    switch (n_tok) { \
        STRATA_MMVF_WIDE(N) \
        case 1: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 1, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok, mmvf_fast()); break; \
        case 2: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 2, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok, mmvf_fast()); break; \
        case 3: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 3, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok, mmvf_fast()); break; \
        case 4: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 4, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok, mmvf_fast()); break; \
        case 5: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 5, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok, mmvf_fast()); break; \
        case 6: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 6, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok, mmvf_fast()); break; \
        default: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 8, false>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok, mmvf_fast()); break; \
    } break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_M(32); STRATA_MMVF_M(64); STRATA_MMVF_M(96); STRATA_MMVF_M(128);
        STRATA_MMVF_M(160); STRATA_MMVF_M(192); STRATA_MMVF_M(224); STRATA_MMVF_M(256);
    }
#undef STRATA_MMVF_M
#undef STRATA_MMVF_WIDE
#ifdef STRATA_MMVF_W
#undef STRATA_MMVF_W
#endif
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    const cudaStream_t st = (cudaStream_t) stream;
#if defined(STRATA_HIP_GFX906)
    if (try_launch_mmvf_wave(x, n_in, w, y, n_out, n_in, n_out, 1, st)) return;
#endif
#define STRATA_MMVF_CASE(N) case N: \
    bf16_f32_mmvf_kernel<N><<<(unsigned) n_out, N, 0, st>>>(x, w, y, (int) n_in, mmvf_fast()); break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_CASE(32);
        STRATA_MMVF_CASE(64);
        STRATA_MMVF_CASE(96);
        STRATA_MMVF_CASE(128);
        STRATA_MMVF_CASE(160);
        STRATA_MMVF_CASE(192);
        STRATA_MMVF_CASE(224);
        STRATA_MMVF_CASE(256);
    }
#undef STRATA_MMVF_CASE
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + cudaGetErrorString(result));
}



void bf16_gemv_fp32_mmvf_cols(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int ncols,
                              void* stream) {
    // columns contiguous: the existing multi-row kernel (weight read once, each output bitwise its one-row call),
    // up to 8 rows per launch
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = ncols - c0 < 8 ? ncols - c0 : 8;
        if (nc == 1) bf16_gemv_fp32_mmvf(x + (size_t) c0 * n_in, w, y + (size_t) c0 * n_out, n_in, n_out, stream);
        else bf16_gemv_fp32_mmvf_multi(x + (size_t) c0 * n_in, n_in, w, y + (size_t) c0 * n_out, n_out, n_in, n_out, nc,
                                       stream);
    }
}

}  // namespace strata::kernels
