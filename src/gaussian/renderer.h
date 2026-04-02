#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "objects.h"
#include "../shaders/shader.h"
#include "../gaussian_utils/gaussian_utils.h"

class Camera;
class Shader;

class GaussianRenderer {
public:
    GaussianRenderer();
    ~GaussianRenderer();

    void init(int width, int height);

    void updateSplats(const std::vector<Splat>& splats);
    void resize(int width, int height);
    void render(
        const Camera& camera,
        int screen_width, int screen_height,
        float scale_modifier = 1.0f
    );
    void save_image(const std::string& filename) const;
    size_t getSplatCount() const;

private:
    Shader* shader = nullptr;

    // CUDA Device Pointers (Splat Data)
    float *d_means3D = nullptr;
    float *d_scales = nullptr;
    float *d_rotations = nullptr;
    float *d_colors = nullptr;
    float *d_opacities = nullptr;
    float *d_cam_params = nullptr;
    
    // CUDA workspace buffers
    char* d_geom_buffer = nullptr;
    char* d_binning_buffer = nullptr;
    char* d_img_buffer = nullptr;
    
    size_t allocated_geom_size = 0;
    size_t allocated_binning_size = 0;
    size_t allocated_img_size = 0;

    std::function<char*(size_t)> geomAlloc;
    std::function<char*(size_t)> binningAlloc;
    std::function<char*(size_t)> imgAlloc;

    // CUDA runtime auxiliary objects
    float* d_bg_color = nullptr;
    float* d_view = nullptr;
    float* d_proj_view = nullptr;
    float* d_cam_pos = nullptr;

    // Output and Interop
    GLuint pbo = 0;
    GLuint display_texture = 0;
    GLuint quad_vao = 0;
    cudaGraphicsResource* pbo_resource = nullptr;

    int current_width = 0;
    int current_height = 0;

    size_t splat_count = 0;

    void allocateCudaBuffer(void** ptr, size_t size);
};