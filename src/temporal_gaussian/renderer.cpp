#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "renderer.h"
#include "../camera/camera.h"
#include "../gaussian/gaussian_parser.h"
#include "../cuda_rasterizer/rasterizer_impl.h"

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / Destruction
// ─────────────────────────────────────────────────────────────────────────────

TemporalGaussianRenderer::TemporalGaussianRenderer() {
    geomAlloc = [this](size_t size) {
        if (!d_geom_buffer || size > allocated_geom_size) {
            if (d_geom_buffer) cudaFree(d_geom_buffer);
            cudaMalloc((void**)&d_geom_buffer, size);
            allocated_geom_size = size;
        }
        return d_geom_buffer;
    };

    binningAlloc = [this](size_t size) {
        if (!d_binning_buffer || size > allocated_binning_size) {
            if (d_binning_buffer) cudaFree(d_binning_buffer);
            cudaMalloc((void**)&d_binning_buffer, size);
            allocated_binning_size = size;
        }
        return d_binning_buffer;
    };

    imgAlloc = [this](size_t size) {
        if (!d_img_buffer || size > allocated_img_size) {
            if (d_img_buffer) cudaFree(d_img_buffer);
            cudaMalloc((void**)&d_img_buffer, size);
            allocated_img_size = size;
        }
        return d_img_buffer;
    };
}

TemporalGaussianRenderer::~TemporalGaussianRenderer() {
    if (d_means3D)    cudaFree(d_means3D);
    if (d_scales)     cudaFree(d_scales);
    if (d_rotations)  cudaFree(d_rotations);
    if (d_colors)     cudaFree(d_colors);
    if (d_opacities)  cudaFree(d_opacities);

    if (d_all_means3D)   cudaFree(d_all_means3D);
    if (d_all_scales)    cudaFree(d_all_scales);
    if (d_all_rotations) cudaFree(d_all_rotations);
    if (d_all_colors)    cudaFree(d_all_colors);
    if (d_all_opacities) cudaFree(d_all_opacities);

    if (d_bg_color)   cudaFree(d_bg_color);
    if (d_cam_params) cudaFree(d_cam_params);

    if (d_geom_buffer)    cudaFree(d_geom_buffer);
    if (d_binning_buffer) cudaFree(d_binning_buffer);
    if (d_img_buffer)     cudaFree(d_img_buffer);

    if (pbo_resource)    cudaGraphicsUnregisterResource(pbo_resource);
    if (pbo)             glDeleteBuffers(1, &pbo);
    if (display_texture) glDeleteTextures(1, &display_texture);
    if (quad_vao)        glDeleteVertexArrays(1, &quad_vao);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Initialisation
// ─────────────────────────────────────────────────────────────────────────────

void TemporalGaussianRenderer::init(int width, int height) {
    shader = std::make_unique<Shader>("src/shaders/gaussian/vertex.glsl",
                                     "src/shaders/gaussian/fragment.glsl");
    glGenVertexArrays(1, &quad_vao);
    resize(width, height);
}

void TemporalGaussianRenderer::allocateCudaBuffer(void** ptr, size_t size) {
    if (*ptr) cudaFree(*ptr);
    cudaMalloc(ptr, size);
}

void TemporalGaussianRenderer::resize(int width, int height) {
    if (width == current_width && height == current_height) return;

    current_width  = width;
    current_height = height;

    if (pbo_resource)    cudaGraphicsUnregisterResource(pbo_resource);
    if (pbo)             glDeleteBuffers(1, &pbo);
    if (display_texture) glDeleteTextures(1, &display_texture);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 (size_t)width * height * 3 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

    glGenTextures(1, &display_texture);
    glBindTexture(GL_TEXTURE_2D, display_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height * 3, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scene loading
// ─────────────────────────────────────────────────────────────────────────────

void TemporalGaussianRenderer::splatsToArrays(const std::vector<Splat>& splats,
                                               std::vector<float>& means3D,
                                               std::vector<float>& scales,
                                               std::vector<float>& rotations,
                                               std::vector<float>& colors,
                                               std::vector<float>& opacities) {
    size_t n = splats.size();
    means3D.resize(n * 3);
    scales.resize(n * 3);
    rotations.resize(n * 4);
    colors.resize(n * 3);
    opacities.resize(n);

    for (size_t i = 0; i < n; i++) {
        const auto& s = splats[i];

        means3D[i*3 + 0] = s.position.x;
        means3D[i*3 + 1] = s.position.y;
        means3D[i*3 + 2] = s.position.z;

        scales[i*3 + 0] = s.scale.x;
        scales[i*3 + 1] = s.scale.y;
        scales[i*3 + 2] = s.scale.z;

        // quaternion stored as w,x,y,z
        rotations[i*4 + 0] = s.rotation.w;
        rotations[i*4 + 1] = s.rotation.x;
        rotations[i*4 + 2] = s.rotation.y;
        rotations[i*4 + 3] = s.rotation.z;

        colors[i*3 + 0] = s.color_dc.x;
        colors[i*3 + 1] = s.color_dc.y;
        colors[i*3 + 2] = s.color_dc.z;

        opacities[i] = s.opacity;
    }
}

void TemporalGaussianRenderer::uploadStaticSplats(const std::vector<Splat>& splats) {
    std::vector<float> means3D, scales, rotations, colors, opacities;
    splatsToArrays(splats, means3D, scales, rotations, colors, opacities);

    // Copy static data into the first half of the combined working buffers.
    cudaMemcpy(d_means3D,   means3D.data(),   means3D.size()   * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales,    scales.data(),    scales.size()    * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rotations, rotations.data(), rotations.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors,    colors.data(),    colors.size()    * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_opacities, opacities.data(), opacities.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void TemporalGaussianRenderer::uploadDynamicFrames(const std::vector<std::vector<Splat>>& frames) {
    // Combined working buffers are already allocated by loadScene.
    // This function only allocates and fills the per-frame storage (GPU or RAM).

    // ── Build flat SOA arrays for all dynamic frames ───────────────────────
    size_t stride3 = dynamic_count * 3;
    size_t stride4 = dynamic_count * 4;
    size_t stride1 = dynamic_count;

    // Reuse temp vectors across frames to avoid repeated allocations
    std::vector<float> m, sc, ro, co, op;

    if (storage_mode == StorageMode::GPU) {
        allocateCudaBuffer((void**)&d_all_means3D,   (size_t)num_frames * stride3 * sizeof(float));
        allocateCudaBuffer((void**)&d_all_scales,    (size_t)num_frames * stride3 * sizeof(float));
        allocateCudaBuffer((void**)&d_all_rotations, (size_t)num_frames * stride4 * sizeof(float));
        allocateCudaBuffer((void**)&d_all_colors,    (size_t)num_frames * stride3 * sizeof(float));
        allocateCudaBuffer((void**)&d_all_opacities, (size_t)num_frames * stride1 * sizeof(float));

        for (int f = 0; f < num_frames; f++) {
            splatsToArrays(frames[f], m, sc, ro, co, op);

            cudaMemcpy(d_all_means3D   + f * stride3, m.data(),  stride3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_scales    + f * stride3, sc.data(), stride3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_rotations + f * stride4, ro.data(), stride4 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_colors    + f * stride3, co.data(), stride3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_opacities + f * stride1, op.data(), stride1 * sizeof(float), cudaMemcpyHostToDevice);
        }
    } else {
        // RAM mode: pack everything into host vectors
        h_all_means3D.resize((size_t)num_frames * stride3);
        h_all_scales.resize((size_t)num_frames * stride3);
        h_all_rotations.resize((size_t)num_frames * stride4);
        h_all_colors.resize((size_t)num_frames * stride3);
        h_all_opacities.resize((size_t)num_frames * stride1);

        for (int f = 0; f < num_frames; f++) {
            splatsToArrays(frames[f], m, sc, ro, co, op);

            std::copy(m.begin(),  m.end(),  h_all_means3D.begin()   + f * stride3);
            std::copy(sc.begin(), sc.end(), h_all_scales.begin()    + f * stride3);
            std::copy(ro.begin(), ro.end(), h_all_rotations.begin() + f * stride4);
            std::copy(co.begin(), co.end(), h_all_colors.begin()    + f * stride3);
            std::copy(op.begin(), op.end(), h_all_opacities.begin() + f * stride1);
        }
    }
}

void TemporalGaussianRenderer::updateDynamicFrame(int frame_idx) {
    if (num_frames == 0) return;
    frame_idx = std::max(0, std::min(frame_idx, num_frames - 1));
    current_frame = frame_idx;

    size_t stride3 = dynamic_count * 3;
    size_t stride4 = dynamic_count * 4;
    size_t stride1 = dynamic_count;

    // Destination: second half of combined buffers
    float* dst_means3D   = d_means3D   + static_count * 3;
    float* dst_scales    = d_scales    + static_count * 3;
    float* dst_rotations = d_rotations + static_count * 4;
    float* dst_colors    = d_colors    + static_count * 3;
    float* dst_opacities = d_opacities + static_count;

    if (storage_mode == StorageMode::GPU) {
        // Device-to-device copy — very fast
        cudaMemcpy(dst_means3D,   d_all_means3D   + frame_idx * stride3, stride3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst_scales,    d_all_scales    + frame_idx * stride3, stride3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst_rotations, d_all_rotations + frame_idx * stride4, stride4 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst_colors,    d_all_colors    + frame_idx * stride3, stride3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst_opacities, d_all_opacities + frame_idx * stride1, stride1 * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        // Host-to-device copy
        cudaMemcpy(dst_means3D,   h_all_means3D.data()   + frame_idx * stride3, stride3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst_scales,    h_all_scales.data()    + frame_idx * stride3, stride3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst_rotations, h_all_rotations.data() + frame_idx * stride4, stride4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst_colors,    h_all_colors.data()    + frame_idx * stride3, stride3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst_opacities, h_all_opacities.data() + frame_idx * stride1, stride1 * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void TemporalGaussianRenderer::loadScene(const std::string& static_ply_path,
                                          const std::string& dynamic_folder_path,
                                          StorageMode mode) {
    storage_mode = mode;

    // ── Collect and sort dynamic PLY files ────────────────────────────────
    std::vector<std::string> dynamic_paths;
    for (const auto& entry : fs::directory_iterator(dynamic_folder_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ply") {
            dynamic_paths.push_back(entry.path().string());
        }
    }
    std::sort(dynamic_paths.begin(), dynamic_paths.end());

    if (dynamic_paths.empty()) {
        std::cerr << "[TemporalGaussianRenderer] No PLY files found in: " << dynamic_folder_path << "\n";
        return;
    }

    std::cout << "[TemporalGaussianRenderer] Loading static scene from: " << static_ply_path << "\n";
    auto static_splats = readGaussianSplats(static_ply_path);
    if (static_splats.empty()) {
        std::cerr << "[TemporalGaussianRenderer] Static PLY is empty or failed to load.\n";
        return;
    }

    std::cout << "[TemporalGaussianRenderer] Loading " << dynamic_paths.size() << " dynamic frames...\n";
    std::vector<std::vector<Splat>> dynamic_frames;
    dynamic_frames.reserve(dynamic_paths.size());

    size_t expected_count = 0;
    for (size_t i = 0; i < dynamic_paths.size(); i++) {
        auto frame = readGaussianSplats(dynamic_paths[i]);
        if (i == 0) {
            expected_count = frame.size();
        } else if (frame.size() != expected_count) {
            std::cerr << "[TemporalGaussianRenderer] Frame " << i << " has " << frame.size()
                      << " splats but expected " << expected_count << ". Skipping.\n";
            continue;
        }
        dynamic_frames.push_back(std::move(frame));
    }

    if (dynamic_frames.empty()) {
        std::cerr << "[TemporalGaussianRenderer] No valid dynamic frames loaded.\n";
        return;
    }

    std::cout << "[TemporalGaussianRenderer] Static: " << static_splats.size()
              << " splats | Dynamic: " << dynamic_frames.size() << " frames x "
              << dynamic_frames[0].size() << " splats\n";

    // ── Allocate combined buffers and fill static portion ─────────────────
    static_count  = static_splats.size();
    dynamic_count = dynamic_frames[0].size();
    num_frames    = static_cast<int>(dynamic_frames.size());
    size_t total  = static_count + dynamic_count;

    allocateCudaBuffer((void**)&d_means3D,   total * 3 * sizeof(float));
    allocateCudaBuffer((void**)&d_scales,    total * 3 * sizeof(float));
    allocateCudaBuffer((void**)&d_rotations, total * 4 * sizeof(float));
    allocateCudaBuffer((void**)&d_colors,    total * 3 * sizeof(float));
    allocateCudaBuffer((void**)&d_opacities, total * 1 * sizeof(float));

    uploadStaticSplats(static_splats);
    uploadDynamicFrames(dynamic_frames);

    // ── Camera parameter buffer ────────────────────────────────────────────
    float bg_color[3] = {0.1f, 0.1f, 0.1f};
    allocateCudaBuffer((void**)&d_bg_color, 3 * sizeof(float));
    cudaMemcpy(d_bg_color, bg_color, 3 * sizeof(float), cudaMemcpyHostToDevice);

    allocateCudaBuffer((void**)&d_cam_params, 35 * sizeof(float));
    d_view      = d_cam_params;
    d_proj_view = d_cam_params + 16;
    d_cam_pos   = d_cam_params + 32;

    // Load frame 0 into the dynamic portion of the working buffers
    current_frame = 0;
    updateDynamicFrame(0);

    std::cout << "[TemporalGaussianRenderer] Scene loaded. Storage: "
              << (mode == StorageMode::GPU ? "GPU" : "RAM") << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Frame control
// ─────────────────────────────────────────────────────────────────────────────

void TemporalGaussianRenderer::setFrame(int frame_idx) {
    if (!isLoaded()) return;
    frame_idx = std::max(0, std::min(frame_idx, num_frames - 1));
    if (frame_idx == current_frame) return;
    updateDynamicFrame(frame_idx);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rendering
// ─────────────────────────────────────────────────────────────────────────────

void TemporalGaussianRenderer::render(const Camera& camera, int width, int height,
                                       float scale_modifier) {
    if (!isLoaded()) return;

    if (width != current_width || height != current_height)
        resize(width, height);

    float aspect_ratio = (float)width / height;
    float fov_y = glm::radians(45.0f);
    float fov_x = 2.0f * std::atan(std::tan(fov_y * 0.5f) * aspect_ratio);
    CameraData cam_data = calculateProjView(camera, fov_x, fov_y);

    float cam_params[35];
    std::memcpy(cam_params,      glm::value_ptr(cam_data.view),     16 * sizeof(float));
    std::memcpy(cam_params + 16, glm::value_ptr(cam_data.proj_view), 16 * sizeof(float));
    std::memcpy(cam_params + 32, glm::value_ptr(cam_data.cam_pos),   3  * sizeof(float));
    cudaMemcpy(d_cam_params, cam_params, 35 * sizeof(float), cudaMemcpyHostToDevice);

    size_t total_count = static_count + dynamic_count;

    float* d_pbo_ptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &num_bytes, pbo_resource);

    CudaRasterizer::Rasterizer::forward(
        geomAlloc, binningAlloc, imgAlloc,
        (int)total_count, 1, 4,
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
        std::tan(fov_x * 0.5f), std::tan(fov_y * 0.5f),
        false,
        d_pbo_ptr,
        nullptr,
        false,
        nullptr,
        false
    );

    cudaGraphicsUnmapResources(1, &pbo_resource, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, display_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height * 3, GL_RED, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    shader->use();
    glUniform1i(shader->getUniform("renderTex"), 0);
    glBindVertexArray(quad_vao);
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEnable(GL_DEPTH_TEST);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Image export
// ─────────────────────────────────────────────────────────────────────────────

void TemporalGaussianRenderer::save_image(const std::string& filename) const {
    int total_pixels   = current_width * current_height;
    int total_elements = total_pixels * 3;

    std::vector<float> h_image(total_elements);

    float* d_pbo_ptr;
    size_t num_bytes;
    cudaGraphicsResource* res = pbo_resource;
    cudaGraphicsMapResources(1, &res, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &num_bytes, res);
    cudaError_t err = cudaMemcpy(h_image.data(), d_pbo_ptr,
                                 total_elements * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    cudaGraphicsUnmapResources(1, &res, 0);

    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << "\n";
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    file << "P3\n" << current_width << " " << current_height << "\n255\n";
    for (int i = 0; i < total_pixels; ++i) {
        float r = std::fmax(0.0f, std::fmin(1.0f, h_image[i]));
        float g = std::fmax(0.0f, std::fmin(1.0f, h_image[total_pixels + i]));
        float b = std::fmax(0.0f, std::fmin(1.0f, h_image[2 * total_pixels + i]));
        file << (int)std::round(r * 255.0f) << " "
             << (int)std::round(g * 255.0f) << " "
             << (int)std::round(b * 255.0f) << "\n";
    }
    file.close();
    std::cout << "Saved frame " << current_frame << " to " << filename << "\n";
}
