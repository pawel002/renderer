#include "cuda_rasterizer.h"
#include "cuda_rasterizer/rasterizer.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>

// Defined in cuda_rasterizer.cu
extern void launchPlanarToRGBA(const float* d_planar, float4* d_rgba, int W, int H);

#define CHECK_CUDA_ERR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
    } \
}

// ---------------------------------------------------------------------------
// CudaRasterizer implementation (host-only methods)
// ---------------------------------------------------------------------------

GaussianRenderer::CudaRasterizer()
    : m_width(0), m_height(0), pbo(0), outputTexture(0), cuda_pbo_resource(nullptr),
      num_splats(0),
      d_means3D(nullptr), d_scales(nullptr), d_rotations(nullptr),
      d_opacities(nullptr), d_colors(nullptr),
      d_geom_buffer(nullptr), allocated_geom_size(0),
      d_binning_buffer(nullptr), allocated_binning_size(0),
      d_img_buffer(nullptr), allocated_image_size(0) {
}

GaussianRenderer::~CudaRasterizer() {
    Destroy();
}

bool GaussianRenderer::Init(int width, int height) {
    m_width = width;
    m_height = height;
    InitGLResources();
    return true;
}

void GaussianRenderer::Destroy() {
    if (cuda_pbo_resource) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        cuda_pbo_resource = nullptr;
    }
    if (pbo) { glDeleteBuffers(1, &pbo); pbo = 0; }
    if (outputTexture) { glDeleteTextures(1, &outputTexture); outputTexture = 0; }

    if (d_means3D) cudaFree(d_means3D);
    if (d_scales) cudaFree(d_scales);
    if (d_rotations) cudaFree(d_rotations);
    if (d_opacities) cudaFree(d_opacities);
    if (d_colors) cudaFree(d_colors);

    if (d_geom_buffer) cudaFree(d_geom_buffer);
    if (d_binning_buffer) cudaFree(d_binning_buffer);
    if (d_img_buffer) cudaFree(d_img_buffer);
}

void GaussianRenderer::InitGLResources() {
    glGenTextures(1, &outputTexture);
    glBindTexture(GL_TEXTURE_2D, outputTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(float) * 4, nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CHECK_CUDA_ERR(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
}

void GaussianRenderer::Resize(int width, int height) {
    if (width == m_width && height == m_height) return;
    if (cuda_pbo_resource) cudaGraphicsUnregisterResource(cuda_pbo_resource);

    m_width = width;
    m_height = height;

    glBindTexture(GL_TEXTURE_2D, outputTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, nullptr);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(float) * 4, nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CHECK_CUDA_ERR(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
}

// ---------------------------------------------------------------------------
// UpdateSplatData – decompose AoS Splat vector into SoA device arrays
// ---------------------------------------------------------------------------
void GaussianRenderer::UpdateSplatData(const std::vector<Splat>& splats) {
    num_splats = splats.size();
    if (num_splats == 0) return;

    // Free previous allocations
    if (d_means3D)   { cudaFree(d_means3D);   d_means3D = nullptr; }
    if (d_scales)    { cudaFree(d_scales);    d_scales = nullptr; }
    if (d_rotations) { cudaFree(d_rotations); d_rotations = nullptr; }
    if (d_opacities) { cudaFree(d_opacities); d_opacities = nullptr; }
    if (d_colors)    { cudaFree(d_colors);    d_colors = nullptr; }

    // Host staging buffers
    std::vector<float> h_means3D(num_splats * 3);
    std::vector<float> h_scales(num_splats * 3);
    std::vector<float> h_rotations(num_splats * 4);
    std::vector<float> h_opacities(num_splats);
    std::vector<float> h_colors(num_splats * 3);

    for (size_t i = 0; i < num_splats; ++i) {
        const Splat& s = splats[i];

        h_means3D[i * 3 + 0] = s.position.x;
        h_means3D[i * 3 + 1] = s.position.y;
        h_means3D[i * 3 + 2] = s.position.z;

        h_scales[i * 3 + 0] = s.scale.x;
        h_scales[i * 3 + 1] = s.scale.y;
        h_scales[i * 3 + 2] = s.scale.z;

        // glm::quat stores (w,x,y,z) internally. The rasterizer expects (w,x,y,z).
        h_rotations[i * 4 + 0] = s.rotation.w;
        h_rotations[i * 4 + 1] = s.rotation.x;
        h_rotations[i * 4 + 2] = s.rotation.y;
        h_rotations[i * 4 + 3] = s.rotation.z;

        h_opacities[i] = s.opacity;

        h_colors[i * 3 + 0] = s.color_dc.x;
        h_colors[i * 3 + 1] = s.color_dc.y;
        h_colors[i * 3 + 2] = s.color_dc.z;
    }

    CHECK_CUDA_ERR(cudaMalloc(&d_means3D,   num_splats * 3 * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_scales,    num_splats * 3 * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_rotations, num_splats * 4 * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_opacities, num_splats *     sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_colors,    num_splats * 3 * sizeof(float)));

    CHECK_CUDA_ERR(cudaMemcpy(d_means3D,   h_means3D.data(),   num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_scales,    h_scales.data(),    num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_rotations, h_rotations.data(), num_splats * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_opacities, h_opacities.data(), num_splats *     sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_colors,    h_colors.data(),    num_splats * 3 * sizeof(float), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Buffer allocation helper
// ---------------------------------------------------------------------------
void GaussianRenderer::AllocateBuffer(char** buffer, size_t& allocated_size, size_t required_size) {
    if (required_size > allocated_size) {
        if (*buffer) cudaFree(*buffer);
        allocated_size = static_cast<size_t>(required_size * 1.2);
        CHECK_CUDA_ERR(cudaMalloc(buffer, allocated_size));
    }
}

// ---------------------------------------------------------------------------
// Render – bridge to ::GaussianRenderer::Rasterizer::forward()
// ---------------------------------------------------------------------------
void GaussianRenderer::Render(const glm::mat4& viewMatrix, const glm::mat4& projMatrix,
                            int screenWidth, int screenHeight) {
    if (num_splats == 0) return;

    // Flatten matrices (GLM stores column-major)
    float viewmat[16];
    float projmat[16];
    memcpy(viewmat, glm::value_ptr(viewMatrix), 16 * sizeof(float));
    memcpy(projmat, glm::value_ptr(projMatrix), 16 * sizeof(float));

    // Upload matrices to device
    float* d_viewmat = nullptr;
    float* d_projmat = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&d_viewmat, 16 * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_projmat, 16 * sizeof(float)));
    CHECK_CUDA_ERR(cudaMemcpy(d_viewmat, viewmat, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_projmat, projmat, 16 * sizeof(float), cudaMemcpyHostToDevice));

    // Camera position
    glm::vec3 camPos = glm::vec3(glm::inverse(viewMatrix)[3]);
    float* d_camPos = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&d_camPos, 3 * sizeof(float)));
    CHECK_CUDA_ERR(cudaMemcpy(d_camPos, glm::value_ptr(camPos), 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Background color
    float bg[3] = { 0.1f, 0.1f, 0.1f };
    float* d_background = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&d_background, 3 * sizeof(float)));
    CHECK_CUDA_ERR(cudaMemcpy(d_background, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

    // FOV calculations (must match the projection matrix used in renderer.cpp)
    float fovY = glm::radians(45.0f);
    float aspect = (float)screenWidth / (float)screenHeight;
    float fovX = 2.0f * atan(tan(fovY / 2.0f) * aspect);
    float tan_fovx = tan(fovX / 2.0f);
    float tan_fovy = tan(fovY / 2.0f);

    // Output color buffer (planar RGB)
    float* d_out_color = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&d_out_color, 3 * screenWidth * screenHeight * sizeof(float)));

    // Allocator lambdas for the rasterizer
    auto geomAlloc = [this](size_t N) -> char* {
        AllocateBuffer(&d_geom_buffer, allocated_geom_size, N);
        return d_geom_buffer;
    };
    auto binAlloc = [this](size_t N) -> char* {
        AllocateBuffer(&d_binning_buffer, allocated_binning_size, N);
        return d_binning_buffer;
    };
    auto imgAlloc = [this](size_t N) -> char* {
        AllocateBuffer(&d_img_buffer, allocated_image_size, N);
        return d_img_buffer;
    };

    // Call the official rasterizer forward pass
    int P = static_cast<int>(num_splats);

    ::CudaRasterizer::Rasterizer::forward(
        geomAlloc,
        binAlloc,
        imgAlloc,
        P, 0, 0,           // P, D=0 (no SH), M=0
        d_background,
        screenWidth, screenHeight,
        d_means3D,
        nullptr,            // shs (null – using precomputed colors)
        d_colors,           // colors_precomp
        d_opacities,
        d_scales,
        1.0f,               // scale_modifier
        d_rotations,
        nullptr,            // cov3D_precomp (compute from scales/rotations)
        d_viewmat,
        d_projmat,
        d_camPos,
        tan_fovx, tan_fovy,
        false,              // prefiltered
        d_out_color,
        nullptr,            // depth (not needed)
        false               // antialiasing
    );

    // Map PBO and convert planar RGB to interleaved RGBA
    float4* d_pbo_ptr = nullptr;
    size_t pbo_size = 0;
    CHECK_CUDA_ERR(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    CHECK_CUDA_ERR(cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &pbo_size, cuda_pbo_resource));

    launchPlanarToRGBA(d_out_color, d_pbo_ptr, screenWidth, screenHeight);

    CHECK_CUDA_ERR(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // Update the OpenGL texture from the PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, outputTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screenWidth, screenHeight, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Cleanup per-frame temporaries
    cudaFree(d_viewmat);
    cudaFree(d_projmat);
    cudaFree(d_camPos);
    cudaFree(d_background);
    cudaFree(d_out_color);
}
