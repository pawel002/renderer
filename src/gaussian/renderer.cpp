#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>

#include "objects.h"
#include "renderer.h"
#include "../camera/camera.h"
#include "../cuda_rasterizer/rasterizer_impl.h"

GaussianRenderer::GaussianRenderer() { }

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

void GaussianRenderer::allocateCudaBuffer(void** ptr, size_t size) {
    if (*ptr) cudaFree(*ptr);
    cudaMalloc((void**)ptr, size);
}

void GaussianRenderer::resize(int width, int height) {
    if (width == current_width && height == current_height) return;
    
    current_width = width;
    current_height = height;

    if (pbo_resource) cudaGraphicsUnregisterResource(pbo_resource);
    if (pbo) glDeleteBuffers(1, &pbo);
    if (display_texture) glDeleteTextures(1, &display_texture);
    if (d_out_color) cudaFree(d_out_color);

    cudaMalloc(&d_out_color, width * height * 3 * sizeof(float));

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

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
    std::vector<float> opacities(splat_count);
    std::vector<float> colors(splat_count * 3);

    for (int i = 0; i < splat_count; i++) {
        const auto& s = splats[i];
        
        means3D[i*3 + 0] = s.position.x; 
        means3D[i*3 + 1] = s.position.y; 
        means3D[i*3 + 2] = s.position.z;
        
        scales[i*3 + 0] = s.scale.x; 
        scales[i*3 + 1] = s.scale.y; 
        scales[i*3 + 2] = s.scale.z;
        
        rotations[i*4 + 0] = s.rotation.w; 
        rotations[i*4 + 1] = s.rotation.x; 
        rotations[i*4 + 2] = s.rotation.y; 
        rotations[i*4 + 3] = s.rotation.z;
        
        opacities[i] = s.opacity;

        colors[i*3 + 0] = s.color_dc.x;
        colors[i*3 + 1] = s.color_dc.y;
        colors[i*3 + 2] = s.color_dc.z;
    }

    std::cout << splats[0] << std::endl;

    // Allocate and copy data to CUDA
    allocateCudaBuffer((void**)&d_means3D, means3D.size() * sizeof(float));
    cudaMemcpy(d_means3D, means3D.data(), means3D.size() * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer((void**)&d_scales, scales.size() * sizeof(float));
    cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer((void**)&d_rotations, rotations.size() * sizeof(float));
    cudaMemcpy(d_rotations, rotations.data(), rotations.size() * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer((void**)&d_opacities, opacities.size() * sizeof(float));
    cudaMemcpy(d_opacities, opacities.data(), opacities.size() * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer((void**)&d_colors, colors.size() * sizeof(float));
    cudaMemcpy(d_colors, colors.data(), colors.size() * sizeof(float), cudaMemcpyHostToDevice);

    // here is the color variable
    float bg_color[3] = {0.1f, 0.1f, 0.1f};
    allocateCudaBuffer((void**)&d_bg_color, 3 * sizeof(float));
    cudaMemcpy(d_bg_color, bg_color, 3 * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer((void**)&d_view, 16 * sizeof(float));
    allocateCudaBuffer((void**)&d_proj_view, 16 * sizeof(float));
    allocateCudaBuffer((void**)&d_cam_pos, 3 * sizeof(float));
}

void GaussianRenderer::render(const Camera& camera, int width, int height, float scale_modifier) {
    if (splat_count == 0) return;

    if (width != current_width || height != current_height) {
        resize(width, height);
    }

    float aspect_ratio = (float)width / height;
    float fov_y = glm::radians(45.0f);
    float fov_x = 2.0f * std::atan(std::tan(fov_y * 0.5f) * aspect_ratio);
    CameraData cam_data = calculateProjView(camera, fov_x, fov_y);
    
    cudaMemcpy(d_view, glm::value_ptr(cam_data.view), 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_view, glm::value_ptr(cam_data.proj_view), 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cam_pos, glm::value_ptr(cam_data.cam_pos), 3 * sizeof(float), cudaMemcpyHostToDevice);

    auto geomAlloc = [&](size_t size) { if (d_geom_buffer) cudaFree(d_geom_buffer); cudaMalloc((void**)&d_geom_buffer, size); return d_geom_buffer; };
    auto binningAlloc = [&](size_t size) { if (d_binning_buffer) cudaFree(d_binning_buffer); cudaMalloc((void**)&d_binning_buffer, size); return d_binning_buffer; };
    auto imgAlloc = [&](size_t size) { if (d_img_buffer) cudaFree(d_img_buffer); cudaMalloc((void**)&d_img_buffer, size); return d_img_buffer; };

    CudaRasterizer::Rasterizer::forward(
        geomAlloc, binningAlloc, imgAlloc, 
        splat_count, 1, 4, 
        d_bg_color, 
        width, height, 
        d_means3D,
        nullptr, 
        d_colors, 
        d_opacities, 
        d_scales, 
        scale_modifier, 
        d_rotations, 
        nullptr,
        d_view, 
        d_proj_view, 
        d_cam_pos,
        tan(fov_x * 0.5f), tan(fov_y * 0.5f),  
        false, 
        d_out_color, 
        nullptr, 
        false, 
        nullptr, 
        true
    );

    float* d_pbo_ptr;
    size_t num_bytes;
    
    cudaGraphicsMapResources(1, &pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &num_bytes, pbo_resource);
    cudaMemcpy(d_pbo_ptr, d_out_color, width * height * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &pbo_resource, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, display_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height * 3, GL_RED, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    shader->use();
    glUniform1i(glGetUniformLocation(shader->ID, "renderTex"), 0);
    
    glBindVertexArray(quad_vao);
    glDisable(GL_DEPTH_TEST); 
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEnable(GL_DEPTH_TEST);
}

CameraData calculateProjView(const Camera& camera, float fov_x, float fov_y, float znear, float zfar) {
    glm::mat4 flipYZ = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));
    glm::mat4 view = flipYZ * camera.getViewMatrix();

    glm::mat4 proj = glm::mat4(0.0f);
    proj[0][0] = 1.0f / std::tan(fov_x * 0.5f);
    proj[1][1] = 1.0f / std::tan(fov_y * 0.5f);
    proj[2][2] = (zfar + znear) / (zfar - znear);
    proj[2][3] = 1.0f;
    proj[3][2] = -(2.0f * znear * zfar) / (zfar - znear);

    return { view, proj * view, camera.position };
}

void GaussianRenderer::save_image(const std::string& filename) const {
    int total_pixels = current_width * current_height;
    int total_elements = total_pixels * 3;
    size_t byte_size = total_elements * sizeof(float);

    // 1. Allocate host memory to receive the float data
    std::vector<float> h_image(total_elements);

    // 2. Copy data from GPU to CPU
    cudaError_t err = cudaMemcpy(h_image.data(), d_out_color, byte_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 3. Open the output file as a standard text file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // 4. Write the P3 header (P3 = ASCII plain text RGB)
    file << "P3\n" << current_width << " " << current_height << "\n255\n";

    // 5. Convert [0.0, 1.0] floats to [0, 255] integers and write as text
    for (int i = 0; i < total_pixels; ++i) {
        // Extract using planar [3, H, W] indexing
        // R is in the first block, G in the second, B in the third
        float r_f = std::fmax(0.0f, std::fmin(1.0f, h_image[i]));
        float g_f = std::fmax(0.0f, std::fmin(1.0f, h_image[total_pixels + i]));
        float b_f = std::fmax(0.0f, std::fmin(1.0f, h_image[2 * total_pixels + i]));

        // Round to nearest integer
        int r = static_cast<int>(std::round(r_f * 255.0f));
        int g = static_cast<int>(std::round(g_f * 255.0f));
        int b = static_cast<int>(std::round(b_f * 255.0f));

        // Write normal numbers separated by spaces, with a newline per pixel
        file << r << " " << g << " " << b << "\n";
    }

    file.close();
    std::cout << "Successfully saved ASCII PPM image to " << filename << std::endl;
}