#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"
#include "../camera/camera.h"
#include "../cuda_rasterizer/rasterizer_impl.h"

GaussianRenderer::GaussianRenderer() :
    shader(nullptr) { }

GaussianRenderer::~GaussianRenderer() {
    if (shader) delete shader;

    if (d_means3D) cudaFree(d_means3D);
    if (d_scales) cudaFree(d_scales);
    if (d_rotations) cudaFree(d_rotations);
    if (d_colors) cudaFree(d_colors);
    if (d_opacities) cudaFree(d_opacities);
    if (d_geom_buffer) cudaFree(d_geom_buffer);
    if (d_binning_buffer) cudaFree(d_binning_buffer);
    if (d_img_buffer) cudaFree(d_img_buffer);
    if (d_out_color) cudaFree(d_out_color);

    // Cleanup OpenGL
    if (pbo_resource) cudaGraphicsUnregisterResource(pbo_resource);
    if (pbo) glDeleteBuffers(1, &pbo);
    if (display_texture) glDeleteTextures(1, &display_texture);
    if (quad_vao) glDeleteVertexArrays(1, &quad_vao);
}

void GaussianRenderer::init(int width, int height) {
    shader = new Shader("src/shaders/gaussian/vertex.glsl", "src/shaders/gaussian/fragment.glsl");
    glGenVertexArrays(1, &quad_vao);
    resize(width, height);
}

void GaussianRenderer::allocateCudaBuffer(float** ptr, size_t size) {
    if (*ptr) cudaFree(*ptr);
    cudaMalloc((void**)ptr, size);
}

void GaussianRenderer::resize(int width, int height) {
    if (width == current_width && height == current_height) return;
    
    current_width = width;
    current_height = height;

    // Clean up old resources if they exist
    if (pbo_resource) cudaGraphicsUnregisterResource(pbo_resource);
    if (pbo) glDeleteBuffers(1, &pbo);
    if (display_texture) glDeleteTextures(1, &display_texture);
    if (d_out_color) cudaFree(d_out_color);

    // 1. Reallocate raw CUDA output buffer
    cudaMalloc(&d_out_color, width * height * 3 * sizeof(float));

    // 2. Create OpenGL Pixel Buffer Object (PBO)
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // 3. Register PBO with CUDA
    cudaGraphicsGLRegisterBuffer(&pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

    // 4. Create OpenGL Texture (Single channel GL_R32F, 3x taller to hold CHW format data)
    glGenTextures(1, &display_texture);
    glBindTexture(GL_TEXTURE_2D, display_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height * 3, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

void GaussianRenderer::updateSplats(const std::vector<Splat>& splats) {
    splat_count = splats.size();
    if (splat_count == 0) return;

    std::vector<float> means3D(splat_count * 3);
    std::vector<float> scales(splat_count * 3);
    std::vector<float> rotations(splat_count * 4);
    std::vector<float> colors(splat_count * 3);
    std::vector<float> opacities(splat_count);

    for (int i = 0; i < splat_count; i++) {
        const auto& s = splats[i];
        
        means3D[i*3 + 0] = s.position.x; means3D[i*3 + 1] = s.position.y; means3D[i*3 + 2] = s.position.z;
        
        scales[i*3 + 0] = s.scale.x; scales[i*3 + 1] = s.scale.y; scales[i*3 + 2] = s.scale.z;
        
        rotations[i*4 + 0] = s.rotation.w; rotations[i*4 + 1] = s.rotation.x; 
        rotations[i*4 + 2] = s.rotation.y; rotations[i*4 + 3] = s.rotation.z;
        
        colors[i*3 + 0] = s.color_dc.x; colors[i*3 + 1] = s.color_dc.y; colors[i*3 + 2] = s.color_dc.z;
        
        opacities[i] = s.opacity;
    }

    allocateCudaBuffer(&d_means3D, means3D.size() * sizeof(float));
    cudaMemcpy(d_means3D, means3D.data(), means3D.size() * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer(&d_scales, scales.size() * sizeof(float));
    cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer(&d_rotations, rotations.size() * sizeof(float));
    cudaMemcpy(d_rotations, rotations.data(), rotations.size() * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer(&d_colors, colors.size() * sizeof(float));
    cudaMemcpy(d_colors, colors.data(), colors.size() * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer(&d_opacities, opacities.size() * sizeof(float));
    cudaMemcpy(d_opacities, opacities.data(), opacities.size() * sizeof(float), cudaMemcpyHostToDevice);

    // here is the color variable
    float bg_color[3] = {0.1f, 0.1f, 0.1f};
    allocateCudaBuffer(&d_bg_color, 3 * sizeof(float));
    cudaMemcpy(d_bg_color, bg_color, 3 * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer(&d_view, 16 * sizeof(float));
    allocateCudaBuffer(&d_proj_view, 16 * sizeof(float));
    allocateCudaBuffer(&d_cam_pos, 3 * sizeof(float));
}

void GaussianRenderer::render(Camera& camera, int width, int height) {
    if (splat_count == 0) return;

    // Automatically catch window resizes without cluttering the main logic
    if (width != current_width || height != current_height) {
        resize(width, height);
    }

    // ==========================================
    // 1. PREPARE CAMERA & MATRICES 
    // ==========================================
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)width / height, 0.1f, 1000.0f);
    
    float fov_y = glm::radians(45.0f);
    float fov_x = 2.0f * atan(tan(fov_y * 0.5f) * ((float)width / height));

    glm::mat4 proj_view = proj * view;
    glm::vec3 cam_pos = camera.position;
    
    cudaMemcpy(d_view, glm::value_ptr(view), 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_view, glm::value_ptr(proj_view), 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cam_pos, glm::value_ptr(cam_pos), 3 * sizeof(float), cudaMemcpyHostToDevice);

    auto geomAlloc = [&](size_t size) { if (d_geom_buffer) cudaFree(d_geom_buffer); cudaMalloc((void**)&d_geom_buffer, size); return d_geom_buffer; };
    auto binningAlloc = [&](size_t size) { if (d_binning_buffer) cudaFree(d_binning_buffer); cudaMalloc((void**)&d_binning_buffer, size); return d_binning_buffer; };
    auto imgAlloc = [&](size_t size) { if (d_img_buffer) cudaFree(d_img_buffer); cudaMalloc((void**)&d_img_buffer, size); return d_img_buffer; };

    // ==========================================
    // 2. RUN CUDA RASTERIZER
    // ==========================================
    CudaRasterizer::Rasterizer::forward(
        geomAlloc, binningAlloc, imgAlloc, splat_count, 0, 0, d_bg_color, width, height, d_means3D,
        nullptr, d_colors, d_opacities, d_scales, 1.0f, d_rotations, nullptr,
        d_view, d_proj_view, d_cam_pos,
        tan(fov_x * 0.5f), tan(fov_y * 0.5f), false, d_out_color, nullptr, false, nullptr, false
    );

    // ==========================================
    // 3. DISPLAY TO OPENGL
    // ==========================================
    float* d_pbo_ptr;
    size_t num_bytes;
    
    // Map, copy, and unmap
    cudaGraphicsMapResources(1, &pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &num_bytes, pbo_resource);
    cudaMemcpy(d_pbo_ptr, d_out_color, width * height * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &pbo_resource, 0);

    // Update OpenGL texture from the PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, display_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height * 3, GL_RED, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Draw the full screen quad!
    shader->use();
    glUniform1i(glGetUniformLocation(shader->ID, "renderTex"), 0);
    
    glBindVertexArray(quad_vao);
    glDisable(GL_DEPTH_TEST); 
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEnable(GL_DEPTH_TEST);
}