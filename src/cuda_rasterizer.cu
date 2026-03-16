#include "cuda_rasterizer.h"
#include "cuda_rasterizer/rasterizer.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define CHECK_CUDA_ERR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
    } \
}

// ---------------------------------------------------------------------------
// Kernel: convert planar RGB (R plane, G plane, B plane) -> interleaved RGBA
// ---------------------------------------------------------------------------
__global__ void planarToInterleavedRGBA(const float* __restrict__ planarRGB,
                                        float4* __restrict__ outRGBA,
                                        int W, int H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (idx >= total) return;

    float r = planarRGB[0 * total + idx];
    float g = planarRGB[1 * total + idx];
    float b = planarRGB[2 * total + idx];
    outRGBA[idx] = make_float4(r, g, b, 1.0f);
}

// ---------------------------------------------------------------------------
// Free function: launch the conversion kernel (called from host code)
// ---------------------------------------------------------------------------
void launchPlanarToRGBA(const float* d_planar, float4* d_rgba, int W, int H) {
    int total = W * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    planarToInterleavedRGBA<<<blocks, threads>>>(d_planar, d_rgba, W, H);
}
