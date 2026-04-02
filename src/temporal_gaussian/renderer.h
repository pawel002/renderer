#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "../gaussian/objects.h"
#include "../shaders/shader.h"
#include "../gaussian_utils/gaussian_utils.h"

class Camera;

class TemporalGaussianRenderer {
public:
    // GPU  – all dynamic frames reside on the GPU (fast playback, high VRAM usage)
    // RAM  – dynamic frames reside in host RAM; one H2D copy per frame change
    enum class StorageMode { GPU, RAM };

    TemporalGaussianRenderer();
    ~TemporalGaussianRenderer();

    void init(int width, int height);

    // Load static splat from `static_ply_path` and all dynamic frames from PLY
    // files found (sorted by name) inside `dynamic_folder_path`.
    void loadScene(const std::string& static_ply_path,
                   const std::string& dynamic_folder_path,
                   StorageMode mode = StorageMode::GPU);

    void resize(int width, int height);
    void render(const Camera& camera, int screen_width, int screen_height,
                float scale_modifier = 1.0f);
    void save_image(const std::string& filename) const;

    void setFrame(int frame_idx);
    int  getCurrentFrame()      const { return current_frame; }
    int  getFrameCount()        const { return num_frames; }
    size_t getStaticCount()     const { return static_count; }
    size_t getDynamicCount()    const { return dynamic_count; }
    size_t getTotalSplatCount() const { return static_count + dynamic_count; }
    bool   isLoaded()           const { return num_frames > 0; }

private:
    std::unique_ptr<Shader> shader;

    StorageMode storage_mode = StorageMode::GPU;

    size_t static_count  = 0;   // number of static splats
    size_t dynamic_count = 0;   // number of dynamic splats per frame
    int    num_frames    = 0;
    int    current_frame = 0;

    // ── Combined working buffers on GPU ────────────────────────────────────
    // Layout: [static_count | dynamic_count]
    // The static portion is filled once; dynamic portion is swapped per frame.
    float* d_means3D    = nullptr;   // (static_count + dynamic_count) * 3
    float* d_scales     = nullptr;   // (static_count + dynamic_count) * 3
    float* d_rotations  = nullptr;   // (static_count + dynamic_count) * 4
    float* d_colors     = nullptr;   // (static_count + dynamic_count) * 3
    float* d_opacities  = nullptr;   // (static_count + dynamic_count) * 1

    // ── GPU storage: all dynamic frames resident on device ─────────────────
    float* d_all_means3D   = nullptr;   // num_frames * dynamic_count * 3
    float* d_all_scales    = nullptr;
    float* d_all_rotations = nullptr;   // num_frames * dynamic_count * 4
    float* d_all_colors    = nullptr;
    float* d_all_opacities = nullptr;

    // ── RAM storage: all dynamic frames in host memory ─────────────────────
    std::vector<float> h_all_means3D;
    std::vector<float> h_all_scales;
    std::vector<float> h_all_rotations;
    std::vector<float> h_all_colors;
    std::vector<float> h_all_opacities;

    // ── Camera / rendering parameters ──────────────────────────────────────
    float* d_bg_color  = nullptr;
    float* d_cam_params = nullptr;
    float* d_view      = nullptr;   // alias into d_cam_params
    float* d_proj_view = nullptr;
    float* d_cam_pos   = nullptr;

    // ── CUDA workspace (geometry / binning / image) ────────────────────────
    char*  d_geom_buffer    = nullptr;
    char*  d_binning_buffer = nullptr;
    char*  d_img_buffer     = nullptr;
    size_t allocated_geom_size    = 0;
    size_t allocated_binning_size = 0;
    size_t allocated_img_size     = 0;
    std::function<char*(size_t)> geomAlloc;
    std::function<char*(size_t)> binningAlloc;
    std::function<char*(size_t)> imgAlloc;

    // ── GL-CUDA interop ────────────────────────────────────────────────────
    GLuint pbo             = 0;
    GLuint display_texture = 0;
    GLuint quad_vao        = 0;
    cudaGraphicsResource* pbo_resource = nullptr;

    int current_width  = 0;
    int current_height = 0;

    // ── Helpers ────────────────────────────────────────────────────────────
    void allocateCudaBuffer(void** ptr, size_t size);
    void uploadStaticSplats(const std::vector<Splat>& splats);
    void uploadDynamicFrames(const std::vector<std::vector<Splat>>& frames);
    void updateDynamicFrame(int frame_idx);

    // Pack a vector of Splat structs into flat SOA float arrays.
    static void splatsToArrays(const std::vector<Splat>& splats,
                               std::vector<float>& means3D,
                               std::vector<float>& scales,
                               std::vector<float>& rotations,
                               std::vector<float>& colors,
                               std::vector<float>& opacities);
};
