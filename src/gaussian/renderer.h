#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "objects.h"
#include "../shaders/shader.h"

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
        int screen_width, int screen_height
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
    float *d_invdepth = nullptr;
    float *d_conv3d = nullptr;
    int *d_radii = nullptr;
    
    // CUDA workspace buffers
    char* d_geom_buffer = nullptr;
    char* d_binning_buffer = nullptr;
    char* d_img_buffer = nullptr;

    // CUDA runtime auxiliary objects
    float* d_bg_color = nullptr;
    float* d_view = nullptr;
    float* d_proj_view = nullptr;
    float* d_cam_pos = nullptr;

    // Output and Interop
    float *d_out_color = nullptr;
    GLuint pbo = 0;
    GLuint display_texture = 0;
    GLuint quad_vao = 0;
    cudaGraphicsResource* pbo_resource = nullptr;

    int current_width = 0;
    int current_height = 0;

    size_t splat_count = 0;

    void allocateCudaBuffer(void** ptr, size_t size);
};

CameraData calculateProjView(const Camera& camera, float fov_x, float fov_y, float znear = 0.01f, float zfar = 100.0f);