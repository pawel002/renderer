#include <vector>
#include <glm/glm.hpp>

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
        Camera& camera,
        int screen_width, int screen_height
    );

    size_t getSplatCount() const;

private:
    Shader* shader;

    // CUDA Device Pointers (Splat Data)
    float *d_means3D = nullptr;
    float *d_scales = nullptr;
    float *d_rotations = nullptr;
    float *d_colors = nullptr;
    float *d_opacities = nullptr;
    
    // CUDA workspace buffers
    char* d_geom_buffer = nullptr;
    char* d_binning_buffer = nullptr;
    char* d_img_buffer = nullptr;

    // Output and Interop
    float *d_out_color = nullptr;
    GLuint pbo = 0;
    GLuint display_texture = 0;
    GLuint quad_vao = 0;
    cudaGraphicsResource* pbo_resource = nullptr;

    int current_width = 0;
    int current_height = 0;

    size_t splat_count;

    void allocateCudaBuffer(float** ptr, size_t size);
};