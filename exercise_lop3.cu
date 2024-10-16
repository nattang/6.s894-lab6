#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>
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

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 1: Inline PTX Warm-Up Exercise

__global__ void
lop3_kernel(uint32_t const *a, uint32_t const *b, uint32_t const *c, uint32_t *out) {
    // TODO: your GPU code here (using inline PTX)
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

void lop3_launch(uint32_t const *a, uint32_t const *b, uint32_t const *c, uint32_t *out) {
    lop3_kernel<<<1, 1>>>(a, b, c, out);
    CUDA_CHECK(cudaGetLastError());
}

void lop3_ref_cpu(
    uint32_t const *a,
    uint32_t const *b,
    uint32_t const *c,
    uint32_t *out) {
    *out = (*a & *b) | *c;
}

void print_binary(uint32_t x) {
    for (int32_t i = 0; i < 32; i++) {
        printf("%d", (x >> (31 - i)) & 1);
    }
}

bool test_lop3() {
    uint32_t *buf_device;
    CUDA_CHECK(cudaMalloc(&buf_device, sizeof(uint32_t) * 4));
    uint32_t *a_device = buf_device;
    uint32_t *b_device = buf_device + 1;
    uint32_t *c_device = buf_device + 2;
    uint32_t *out_device = buf_device + 3;

    auto test_cases = std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1},
        // Inputs generated uniformly at random:
        {0b11011001010100110101001001111010U,
         0b10001101001011010110001100011101U,
         0b1100000110110111001110110011111U},
        {0b101110000010001100110010000110U,
         0b1010001000110001011100010000101U,
         0b10111001001010000100001111100110U},
        {0b100010101110000100100101100011U,
         0b11011001100111011111010111000100U,
         0b10100000010101011101001001100101U},
        {0b11111010000111100110111001000000U,
         0b10010101110001100100101100000101U,
         0b1010010110101011100010011111100U},
    };

    for (auto const &test_case : test_cases) {
        uint32_t a = std::get<0>(test_case);
        uint32_t b = std::get<1>(test_case);
        uint32_t c = std::get<2>(test_case);

        uint32_t expected;
        lop3_ref_cpu(&a, &b, &c, &expected);

        CUDA_CHECK(cudaMemcpy(a_device, &a, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_device, &b, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c_device, &c, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(out_device, 0, sizeof(uint32_t)));

        lop3_launch(a_device, b_device, c_device, out_device);

        uint32_t actual;
        CUDA_CHECK(
            cudaMemcpy(&actual, out_device, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if (expected != actual) {
            printf("lop3 test failed:\n");
            printf("  a        = 0b");
            print_binary(a);
            printf("\n");
            printf("  b        = 0b");
            print_binary(b);
            printf("\n");
            printf("  c        = 0b");
            print_binary(c);
            printf("\n");
            printf("  expected = 0b");
            print_binary(expected);
            printf("\n");
            printf("  actual   = 0b");
            print_binary(actual);
            printf("\n");
            return false;
        }
    }

    CUDA_CHECK(cudaFree(buf_device));

    return true;
}

int main(int argc, char const *const *argv) {
    printf("Testing lop3 kernel...\n");
    bool ok = test_lop3();
    if (ok) {
        printf("lop3 tests passed\n");
    }
    return !ok;
}