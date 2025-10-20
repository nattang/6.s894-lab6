#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}
//clang-format off
// __device__ __forceinline__ void async_commit_group() {
//     asm volatile("cp.async.commit_group;\n" ::);
// }

// template <int N> __device__ __forceinline__ void async_wait_pending() {
//     asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
// }
//clang-format on

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

// OPTIONAL: Uncomment this block to include your kernel implementation
// from Lab 5 for easy comparison.

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k (Baseline from Lab 5)

// #define HAS_LAB_5_BASELINE_IMPL // <~~ keep this line if you want to benchmark your Lab
// 5 kernel!

#define TILE_I 128
#define TILE_J 128
#define TILE_K 32 // must be <= TILE_I and TILE_J

#define WARP_TILE_I 64
#define WARP_TILE_J 32

#define NUM_WARPS_X (TILE_J / WARP_TILE_J) // 4
#define NUM_WARPS_Y (TILE_I / WARP_TILE_I) // 2

#define BLOCK_DIM_X 32 // 32
#define BLOCK_DIM_Y 8  // 8

#define MICRO_TILE 8
#define MICRO_TILE_I 8
#define MICRO_TILE_J 8

#define TILE_K_PAD (TILE_K + 4)
#define TILE_J_PAD (TILE_J + 36)

#define A_SHMEM_COLS_PER_THREAD (TILE_K / BLOCK_DIM_X) // 1
#define A_SHMEM_COLS_PER_THREAD_IMP (TILE_K / (TILE_J / MICRO_TILE))

#define A_SHMEM_ROWS_PER_THREAD (TILE_I / BLOCK_DIM_Y)               // 8
#define A_SHMEM_ROWS_PER_THREAD_IMP (TILE_I / (TILE_I / MICRO_TILE)) // 1

#define B_SHMEM_COLS_PER_THREAD (TILE_J / BLOCK_DIM_X)               // 8
#define B_SHMEM_COLS_PER_THREAD_IMP (TILE_J / (TILE_J / MICRO_TILE)) // 1

#define B_SHMEM_ROWS_PER_THREAD (TILE_K / BLOCK_DIM_Y)               // 1
#define B_SHMEM_ROWS_PER_THREAD_IMP (TILE_K / (TILE_I / MICRO_TILE)) // 1

// #define SPLIT_K_TILE_CHUNK 1024
#define SPLIT_K_TILE 1024

////////////////////////////////////////////////////////////////////////////////
// Tensor Core GPU Implementation

// #define BLOCK_DIM_X_TC (TILE_J / MICRO_TILE_J)
// #define BLOCK_DIM_Y_TC (TILE_I / MICRO_TILE_I)

namespace matmul_tensor {
__global__ void __launch_bounds__((TILE_I / MICRO_TILE) * (TILE_J / MICRO_TILE))
    matmul_improved_main(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        int split_k,
        float const *a, /* pointer to GPU memory */
        float const *b, /* pointer to GPU memory */
        float *workspace /* pointer to GPU memory */) {

    int thread_col = threadIdx.x; // 0 to TILE_J / MICROTILE
    int thread_row = threadIdx.y; // 0 to TILE_I / MICROTILE

    extern __shared__ __align__(16) float shared[];
    float *A_shared = shared;                     // [TILE_K][TILE_I]: transposed
    float *B_shared = A_shared + TILE_I * TILE_K; // [TILE_K][TILE_J]

    float c_reg[MICRO_TILE][MICRO_TILE] = {0}; // partial sum in registers
    float a_reg[MICRO_TILE];
    float b_reg[MICRO_TILE];

    int k_per_split = CEIL_DIV(size_k, split_k);
    int global_k_start = k_per_split * blockIdx.z;
    int global_k_end = MIN(global_k_start + k_per_split, size_k);

    for (int global_k = global_k_start; global_k < global_k_end; global_k += TILE_K) {
        // TODO optimize the memory access pattern with stride for coalescing ??
        // each tile loads MICRO_TILE rows and TILE_K / cols_per_thread cols
        int a_shmem_row_start = thread_row * A_SHMEM_ROWS_PER_THREAD_IMP;
        int a_shmem_col_start = thread_col * A_SHMEM_COLS_PER_THREAD_IMP;

        for (int col_offset = 0; col_offset < A_SHMEM_COLS_PER_THREAD_IMP;
             col_offset += 1) {
            int shmem_col = a_shmem_col_start + col_offset;
            int a_col = global_k + shmem_col;
            for (int row_offset = 0; row_offset < A_SHMEM_ROWS_PER_THREAD_IMP;
                 row_offset += 1) {
                int shmem_row = a_shmem_row_start + row_offset;
                int a_row = blockIdx.y * TILE_I + shmem_row;
                // if (a_row < size_i) { // uncomment if  TILE_I not multiple of 16
                A_shared[shmem_col * TILE_I + shmem_row] = a[a_row * size_k + a_col];
                // } else {
                // A_shared[shmem_col * TILE_I + shmem_row] = 0.0f;
                // }
            }
        }

        // load B tile
        int b_shmem_row_start = thread_row * B_SHMEM_ROWS_PER_THREAD_IMP;
        int b_shmem_col_start = thread_col * B_SHMEM_COLS_PER_THREAD_IMP;
        for (int row_offset = 0; row_offset < B_SHMEM_ROWS_PER_THREAD_IMP; ++row_offset) {
            int shmem_row = b_shmem_row_start + row_offset;
            int b_row = global_k + shmem_row;
            for (int col_offset = 0; col_offset < B_SHMEM_COLS_PER_THREAD_IMP;
                 col_offset += 4) {
                int shmem_col = b_shmem_col_start + col_offset;
                int b_col = blockIdx.x * TILE_J + shmem_col;

                if (b_row < size_k && b_col + 3 < size_j) {
                    float4 b_vec =
                        reinterpret_cast<const float4 *>(&b[b_row * size_j + b_col])[0];
                    reinterpret_cast<float4 *>(
                        &B_shared[shmem_row * TILE_J + shmem_col])[0] = b_vec;
                } else {
                    for (int offset = 0; offset < 4; ++offset) {
                        int b_col_offset = b_col + offset;
                        // if (b_row < size_k && b_col_offset < size_j) { // uncomment if
                        // TILE_J not multiple of 16
                        B_shared[shmem_row * TILE_J + b_col_offset] =
                            b[b_row * size_j + b_col_offset];
                        // } else {
                        // B_shared[shmem_row * TILE_J + b_col_offset] = 0.0f;
                        // }
                    }
                }
            }
        }
        __syncthreads();

        for (int k_inner = 0; k_inner < TILE_K; ++k_inner) {
            int row_base = thread_row * MICRO_TILE;
            int col_base = thread_col * MICRO_TILE;

            for (int m = 0; m < MICRO_TILE; ++m) {
                int a_shmem_row = row_base + m;
                a_reg[m] = A_shared[k_inner * TILE_I + a_shmem_row];
            }

            for (int n = 0; n < MICRO_TILE; ++n) {
                int b_shmem_col = col_base + n;
                b_reg[n] = B_shared[k_inner * TILE_J + b_shmem_col];
            }

            for (int m = 0; m < MICRO_TILE; ++m) {
                for (int n = 0; n < MICRO_TILE; ++n) {
                    c_reg[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }
        __syncthreads();
    }

    int workspace_j_start = blockIdx.x * TILE_J + thread_col * MICRO_TILE; // col of c
    int workspace_i_start = blockIdx.y * TILE_I + thread_row * MICRO_TILE; // row of c

    int workspace_plane_size = size_i * size_j;
    int workspace_plane_offset = (blockIdx.z * workspace_plane_size);

    // write to independent workspace c
    for (int m = 0; m < MICRO_TILE; ++m) {
        for (int n = 0; n < MICRO_TILE; n += 4) {
            int row = workspace_i_start + m;
            int col = workspace_j_start + n;

            int workspace_idx = workspace_plane_offset + row * size_j + col;
            // if (row < size_i && col < size_j) {
            //     workspace[workspace_idx] = c_reg[m][n];
            // }
            if (row < size_i && col + 3 < size_j) {
                float4 c_vec = reinterpret_cast<const float4 *>(&c_reg[m][n])[0];
                reinterpret_cast<float4 *>(&workspace[workspace_idx])[0] = c_vec;
            } else {
                for (int offset = 0; offset < 4; ++offset) {
                    int col_offset = col + offset;
                    if (row < size_i && col_offset < size_j) {
                        workspace[workspace_idx + offset] = c_reg[m][n + offset];
                    }
                }
            }
        }
    }
}

__global__ void __launch_bounds__((TILE_I / MICRO_TILE) * (TILE_J / MICRO_TILE))
    matmul_improved_reduce(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        int split_k,
        float const *workspace,
        float *c /* pointer to GPU memory */) {
    int thread_col = threadIdx.x; // 0 to TILE_J / MICROTILE
    int thread_row = threadIdx.y; // 0 to TILE_I / MICROTILE

    int c_j_start = blockIdx.x * TILE_J + thread_col * MICRO_TILE; // col of c
    int c_i_start = blockIdx.y * TILE_I + thread_row * MICRO_TILE; // row of c

    int workspace_j_start = blockIdx.x * TILE_J + thread_col * MICRO_TILE; // col of c
    int workspace_i_start = blockIdx.y * TILE_I + thread_row * MICRO_TILE; // row of c

    int workspace_plane_size = size_i * size_j;
    int workspace_plane_offset = 0;

    float c_reg[MICRO_TILE][MICRO_TILE] = {0}; // partial sum in registers

    // iterate over split_k dimension in workspace
    // TODO: vectorize
    for (int k = 0; k < split_k; ++k) {
        for (int m = 0; m < MICRO_TILE; ++m) {
            for (int n = 0; n < MICRO_TILE; n += 4) {
                int row = workspace_i_start + m;
                int col = workspace_j_start + n;

                int workspace_idx = workspace_plane_offset + row * size_j + col;
                if (row < size_i && col + 3 < size_j) {
                    float4 c_vec =
                        reinterpret_cast<const float4 *>(&workspace[workspace_idx])[0];
                    float4 c_reg_vec = reinterpret_cast<float4 *>(&c_reg[m][n])[0];
                    c_reg_vec.x += c_vec.x;
                    c_reg_vec.y += c_vec.y;
                    c_reg_vec.z += c_vec.z;
                    c_reg_vec.w += c_vec.w;
                    reinterpret_cast<float4 *>(&c_reg[m][n])[0] = c_reg_vec;
                } else {
                    for (int offset = 0; offset < 4; ++offset) {
                        int col_offset = col + offset;
                        if (row < size_i && col_offset < size_j) {
                            c_reg[m][n + offset] += workspace[row * size_j + col_offset];
                        }
                    }
                }
            }
        }
        workspace_plane_offset += workspace_plane_size;
    }

    // write back C
    for (int m = 0; m < MICRO_TILE; ++m) {
        for (int n = 0; n < MICRO_TILE; n += 4) {
            int row = c_i_start + m;
            int col = c_j_start + n;
            if (row < size_i && col + 3 < size_j) {
                reinterpret_cast<float4 *>(&c[row * size_j + col])[0] =
                    reinterpret_cast<float4 *>(&c_reg[m][n])[0];
            } else {
                for (int offset = 0; offset < 4; ++offset) {
                    int col_offset = col + offset;
                    if (row < size_i && col_offset < size_j) {
                        c[row * size_j + col_offset] = c_reg[m][n + offset];
                    }
                }
            }
        }
    }
}

__global__ void matmul_tensor_main(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    int split_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *workspace /* pointer to GPU memory */) {

    int thread_col = threadIdx.x; // 0 to TILE_J / MICROTILE
    int thread_row = threadIdx.y; // 0 to TILE_I / MICROTILE

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    extern __shared__ __align__(16) float shared[];
    float *A_shared = shared;                     // [TILE_I][TILE_K]: untranspose
    float *B_shared = A_shared + TILE_I * TILE_K_PAD; // [TILE_K][TILE_J]

    int k_per_split = CEIL_DIV(size_k, split_k);
    int global_k_start = k_per_split * blockIdx.z;
    int global_k_end = MIN(global_k_start + k_per_split, size_k);

    int warp_row_base = (warp_id / NUM_WARPS_X) * WARP_TILE_I;
    int warp_col_base = (warp_id % NUM_WARPS_X) * WARP_TILE_J;

    // TODO: switch to use static variables
    uint32_t c_reg[4][4][4] = {0}; // partial sum in registers
    uint32_t a_reg[4][4][4];
    uint32_t b_reg[4][4][2];

    for (int global_k = global_k_start; global_k < global_k_end; global_k += TILE_K) {
        // TODO optimize the memory access pattern with stride for coalescing ??
        // each tile loads MICRO_TILE rows and TILE_K / cols_per_thread cols
        int a_shmem_row_start = thread_row * A_SHMEM_ROWS_PER_THREAD;
        int a_shmem_col_start = thread_col * A_SHMEM_COLS_PER_THREAD;

        // load this thread's portion into shared memory
        for (int col_offset = 0; col_offset < A_SHMEM_COLS_PER_THREAD; col_offset += 1) {
            int shmem_col = a_shmem_col_start + col_offset;
            int a_col = global_k + shmem_col;
            for (int row_offset = 0; row_offset < A_SHMEM_ROWS_PER_THREAD;
                 row_offset += 1) {
                int shmem_row = a_shmem_row_start + row_offset;
                int a_row = blockIdx.y * TILE_I + shmem_row;
                // if (a_row < size_i) { // uncomment if  TILE_I not multiple of 16
                A_shared[shmem_row * TILE_K_PAD + shmem_col] = a[a_row * size_k + a_col];
                // } else {
                // A_shared[shmem_col * TILE_I + shmem_row] = 0.0f;
                // }
            }
        }

        // and same for B
        int b_shmem_row_start = thread_row * B_SHMEM_ROWS_PER_THREAD;
        int b_shmem_col_start = thread_col * B_SHMEM_COLS_PER_THREAD;
        for (int row_offset = 0; row_offset < B_SHMEM_ROWS_PER_THREAD; ++row_offset) {
            int shmem_row = b_shmem_row_start + row_offset;
            int b_row = global_k + shmem_row;
            for (int col_offset = 0; col_offset < B_SHMEM_COLS_PER_THREAD;
                 col_offset += 4) {
                int shmem_col = b_shmem_col_start + col_offset;
                int b_col = blockIdx.x * TILE_J + shmem_col;

                if (b_row < size_k && b_col + 3 < size_j) {
                    float4 b_vec =
                        reinterpret_cast<const float4 *>(&b[b_row * size_j + b_col])[0];
                    reinterpret_cast<float4 *>(
                        &B_shared[shmem_row * TILE_J_PAD + shmem_col])[0] = b_vec;
                } else {
                    for (int offset = 0; offset < 4; ++offset) {
                        int b_col_offset = b_col + offset;
                        // if (b_row < size_k && b_col_offset < size_j) { // uncomment if
                        // TILE_J not multiple of 16
                        B_shared[shmem_row * TILE_J_PAD + b_col_offset] =
                            b[b_row * size_j + b_col_offset];
                        // } else {
                        // B_shared[shmem_row * TILE_J + b_col_offset] = 0.0f;
                        // }
                    }
                }
            }
        }
        __syncthreads();

        for (int k_inner = 0; k_inner < TILE_K; k_inner += 8) {
            // split the warp tile into 16x8 micro tiles
            for (int mrow = 0; mrow < 4; ++mrow) {     // microtile row
                for (int mcol = 0; mcol < 4; ++mcol) { // microtile col
                    int a_subtile_row =
                        warp_row_base + (mrow * 16); // 16 rows per subtile
                    int a_subtile_col = k_inner;
                    int a_subtile_base = ((a_subtile_row * TILE_K_PAD) + a_subtile_col);

                    int a1_row = lane_id / 4;
                    int a1_col = lane_id % 4;
                    // load A_shared into subreg

                    a_reg[mrow][mcol][0] = __float_as_uint(
                        A_shared[(a_subtile_base) + (a1_row * TILE_K_PAD) + a1_col]);
                    a_reg[mrow][mcol][1] = __float_as_uint(
                        A_shared[(a_subtile_base) + ((a1_row + 8) * TILE_K_PAD) + a1_col]);
                    a_reg[mrow][mcol][2] = __float_as_uint(
                        A_shared[(a_subtile_base) + (a1_row * TILE_K_PAD) + a1_col + 4]);
                    a_reg[mrow][mcol][3] = __float_as_uint(
                        A_shared
                            [(a_subtile_base) + ((a1_row + 8) * TILE_K_PAD) + a1_col + 4]);

                    int b1_col = lane_id / 4;
                    int b1_row = lane_id % 4;

                    int b_subtile_row = k_inner;
                    int b_subtile_col = warp_col_base + (mcol * 8); // 8 cols per subtile
                    int b_subtile_base = (b_subtile_row * TILE_J_PAD) + b_subtile_col;


                    b_reg[mrow][mcol][0] = __float_as_uint(
                        B_shared[(b_subtile_base) + (b1_row * TILE_J_PAD) + b1_col]);
                    b_reg[mrow][mcol][1] = __float_as_uint(
                        B_shared[(b_subtile_base) + ((b1_row + 4) * TILE_J_PAD) + b1_col]);
                }
            }

            for (int mrow = 0; mrow < 4; ++mrow) { // microtile row
                for (int mcol = 0; mcol < 4; ++mcol) {
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+r"(c_reg[mrow][mcol][0]),
                                   "+r"(c_reg[mrow][mcol][1]),
                                   "+r"(c_reg[mrow][mcol][2]),
                                   "+r"(c_reg[mrow][mcol][3])
                                 : "r"(a_reg[mrow][mcol][0]),
                                   "r"(a_reg[mrow][mcol][1]),
                                   "r"(a_reg[mrow][mcol][2]),
                                   "r"(a_reg[mrow][mcol][3]),
                                   "r"(b_reg[mrow][mcol][0]),
                                   "r"(b_reg[mrow][mcol][1])
                                 : "memory");
                }
            }
        }
        __syncthreads();
    }

    int workspace_j_start = blockIdx.x * TILE_J; // col of c
    int workspace_i_start = blockIdx.y * TILE_I; // row of c

    int workspace_plane_size = size_i * size_j;
    int workspace_plane_offset = (blockIdx.z * workspace_plane_size);

    // write to independent workspace c
    // for each of the subtiles in the warp tile
    int subtile_row = workspace_i_start + warp_row_base;

    for (int m = 0; m < 4; ++m) {
        int subtile_col = workspace_j_start + warp_col_base;
        for (int n = 0; n < 4; ++n) {

            int workspace_idx =
                workspace_plane_offset + (subtile_row)*size_j + (subtile_col);
            int c1_row = lane_id / 4;
            int c1_col = lane_id % 4 * 2;

            // if (c1_row < (unsigned)size_i && c1_col < (unsigned)size_j) {
            *reinterpret_cast<float2 *>(
                &workspace[workspace_idx + (c1_row * size_j) + c1_col]) =
                make_float2(
                    __uint_as_float(c_reg[m][n][0]),
                    __uint_as_float(c_reg[m][n][1]));

            *reinterpret_cast<float2 *>(
                &workspace[workspace_idx + ((c1_row + 8) * size_j) + c1_col]) =
                make_float2(
                    __uint_as_float(c_reg[m][n][2]),
                    __uint_as_float(c_reg[m][n][3]));

            subtile_col += 8;
        }
        subtile_row += 16;
    }
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    int split_k = CEIL_DIV(size_k, SPLIT_K_TILE);
    // int split_k = get_split_k(size_k);
    return (size_t)(split_k * size_i * size_j * sizeof(float));
}

void launch_matmul_tensor(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    int split_k = CEIL_DIV(size_k, SPLIT_K_TILE);
    if (size_i >= 512) {
        dim3 gridDimMain = dim3(CEIL_DIV(size_j, TILE_J), CEIL_DIV(size_i, TILE_I), 1);
        dim3 blockDimMain = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);

        uint32_t shmem_size_bytes =
            (TILE_I * TILE_K_PAD * sizeof(float)) + (TILE_K * TILE_J_PAD * sizeof(float));

        CUDA_CHECK(cudaFuncSetAttribute(
            matmul_tensor_main,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size_bytes));
        matmul_tensor_main<<<gridDimMain, blockDimMain, shmem_size_bytes>>>(
            size_i,
            size_j,
            size_k,
            1,
            a,
            b,
            c);

    } else if (size_i >= 128) {
        dim3 gridDimMain = dim3(CEIL_DIV(size_j, TILE_J), CEIL_DIV(size_i, TILE_I), split_k);
        dim3 blockDimMain = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);

        uint32_t shmem_size_bytes =
            (TILE_I * TILE_K_PAD * sizeof(float)) + (TILE_K * TILE_J_PAD * sizeof(float));

        CUDA_CHECK(cudaFuncSetAttribute(
            matmul_tensor_main,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size_bytes));
        matmul_tensor_main<<<gridDimMain, blockDimMain, shmem_size_bytes>>>(
            size_i,
            size_j,
            size_k,
            split_k,
            a,
            b,
            (float *)workspace);

        dim3 blockDimReduce = dim3((TILE_J / MICRO_TILE), (TILE_I / MICRO_TILE));
        matmul_improved_reduce<<<gridDimMain, blockDimReduce>>>(
            size_i,
            size_j,
            size_k,
            split_k,
            (float *)workspace,
            c);
        

    } else {
        dim3 gridDimMain =
            dim3(CEIL_DIV(size_j, TILE_J), CEIL_DIV(size_i, TILE_I), split_k);
        dim3 blockDimMain = dim3((TILE_J / MICRO_TILE), (TILE_I / MICRO_TILE));

        uint32_t shmem_size_bytes =
            (TILE_I * TILE_K * sizeof(float)) + (TILE_K * TILE_J * sizeof(float));

        CUDA_CHECK(cudaFuncSetAttribute(
            matmul_improved_main,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size_bytes));

        matmul_improved_main<<<gridDimMain, blockDimMain, shmem_size_bytes>>>(
            size_i,
            size_j,
            size_k,
            split_k,
            a,
            b,
            (float *)workspace);

        dim3 gridDimReduce = dim3(CEIL_DIV(size_j, TILE_J), CEIL_DIV(size_i, TILE_I));
        dim3 blockDimReduce = dim3((TILE_J / MICRO_TILE), (TILE_I / MICRO_TILE));

        matmul_improved_reduce<<<gridDimReduce, blockDimReduce>>>(
            size_i,
            size_j,
            size_k,
            split_k,
            (float *)workspace,
            c);
    }
}

}; // namespace matmul_tensor

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_k}) == data.a.end()) {
            data.a[{size_i, size_k}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "x" +
                    std::to_string(size_k) + ".bin",
                size_i * size_k);
        }

        if (data.b.find({size_k, size_j}) == data.b.end()) {
            data.b[{size_k, size_j}] = read_data(
                path_prefix + "b_" + std::to_string(size_k) + "x" +
                    std::to_string(size_j) + ".bin",
                size_k * size_j);
        }

        if (data.c.find({size_i, size_j, size_k}) == data.c.end()) {
            data.c[{size_i, size_j, size_k}] = read_data(
                path_prefix + "c_" + std::to_string(size_i) + "x" +
                    std::to_string(size_j) + "x" + std::to_string(size_k) + ".bin",
                size_i * size_j);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;
    auto size_k = config.size_k;

    auto const &a = data.a.at({size_i, size_k});
    auto const &b = data.b.at({size_k, size_j});
    auto const &c = data.c.at({size_i, size_j, size_k});

    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_k * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        b_gpu,
        b.data(),
        size_k * size_j * sizeof(float),
        cudaMemcpyHostToDevice));

    size_t workspace_size = Impl::get_workspace_size(size_i, size_j, size_k);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024 * 1024 * 64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024 * 1024 * 64));

    if (phase == Phase::BENCHMARK) {
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    } else {
        printf("  warmup %6d  %6d  %6d", size_i, size_j, size_k);
    }

    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);

    std::vector<float> c_out_host(size_i * size_j);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size_i * size_j * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < size_j; ++j) {
            float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
            mse += diff * diff;
            ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
        }
    }
    mse /= size_i * size_j;
    ref_mean_square /= size_i * size_j;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-3) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
            target_time_ms,
            1,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
                CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024 * 1024 * 64));
            },
            [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
            });

        if (phase == Phase::BENCHMARK) {
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size_i, size_j, size_k}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
    CUDA_CHECK(cudaFree(flush_gpu));
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "size_i",
            "size_j",
            "size_k",
            "RRMSE",
            "time (ms)",
            "TFLOP/s");
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "-------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

#ifdef HAS_LAB_5_BASELINE_IMPL

struct MatmulImprovedReduce {
    constexpr static char const *name = "matmul_improved_reduce";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_improved_reduce::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_reduce::launch_matmul_improved_reduce(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

#endif

struct MatmulTensor {
    constexpr static char const *name = "matmul_tensor";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_tensor::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_tensor::launch_matmul_tensor(size_i, size_j, size_k, a, b, c, workspace);
    }
};

BenchmarkResults get_cublas_fma_results() {
    // Hard-coded data collected on A4000 GPU
    return BenchmarkResults{
        "cublas_fma",
        {
            {{3072, 3072, 3072}, 3.152},
            {{2048, 3072, 3072}, 2.174},
            {{1024, 3072, 3072}, 1.090},
            {{512, 3072, 3072}, 0.559},
            {{256, 3072, 3072}, 0.356},
            {{128, 3072, 3072}, 0.256},
            {{64, 3072, 3072}, 0.194},
            {{32, 3072, 3072}, 0.181},
            {{16, 3072, 3072}, 0.181},
        }};
}

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_5_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulTensor>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size_i, size_j, size_k] = config;
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
            file << "      \"size_k\": " << size_k << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

void print_speedup(
    std::vector<BenchmarkConfig> const &configs,
    BenchmarkResults const &first,
    BenchmarkResults const &second) {
    printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
    printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
    printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
        auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
        auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
        if (it_first != first.elapsed_ms.end() && it_second != second.elapsed_ms.end()) {
            printf("  %6.02fx", it_first->second / it_second->second);
        } else {
            printf("  %7s", "-");
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";

    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        {2048, 3072, 3072},
        {1024, 3072, 3072},
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            print_speedup(configs, results.at(i), results.at(j));
        }
    }

    printf("\n-----------------------------------------------------------\n");
    printf("---- Comparison to non-tensor-core cuBLAS performance: ----\n");
    printf("-----------------------------------------------------------\n");

    print_speedup(configs, get_cublas_fma_results(), results.at(results.size() - 1));

    write_json_results("out/results.json", results);

    return 0;
}
