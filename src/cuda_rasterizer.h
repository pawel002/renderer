#pragma once
#include <vector>
#include <functional>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include "splat_parser.h"

class GaussianRenderer {
public:
    GaussianRenderer();
    ~GaussianRenderer();

    bool Init(int width, int height);
    void Destroy();

    void Resize(int width, int height);
    
    // Uploads splats to GPU (cudaMalloc & cudaMemcpy)
    void UpdateSplatData(const std::vector<Splat>& splats);

    // Executes the CUDA rasterization pipeline and outputs to the pixel buffer
    void Render(const glm::mat4& viewMatrix, const glm::mat4& projMatrix, int screenWidth, int screenHeight);

    // Returns the OpenGL texture ID that the CUDA kernel wrote to
    GLuint GetOutputTexture() const { return outputTexture; }

private:
    int m_width;
    int m_height;
    
    GLuint pbo;
    GLuint outputTexture;
    struct cudaGraphicsResource* cuda_pbo_resource;

    size_t num_splats;

    // SoA (Structure of Arrays) for diff-gaussian-rasterization
    float* d_means3D;
    float* d_scales;
    float* d_rotations;
    float* d_opacities;
    float* d_colors;
    
    // Memory pools for the rasterizer's dynamic buffers
    char* d_geom_buffer;
    size_t allocated_geom_size;

    char* d_binning_buffer;
    size_t allocated_binning_size;

    char* d_img_buffer;
    size_t allocated_image_size;

    void InitGLResources();
    void AllocateBuffer(char** buffer, size_t& allocated_size, size_t required_size);
};
