#include <vector>
#include <glm/glm.hpp>

#include "objects.h"
#include "../shaders/shader.h"

class Camera;
class Shader;

class PointCloudRenderer {
public:
    PointCloudRenderer();
    ~PointCloudRenderer();

    void init();

    void updatePointCloudData(const std::vector<Point>& points);
    void updateCameraData(const std::vector<CameraPose>& cameras);

    void render(
        Camera& camera,
        float screen_width, float screen_height,
        float base_point_size, float min_point_size, float max_point_size,
        bool show_cameras    
    );

    size_t getPointCount();
    size_t getPoseCount();

private:
    Shader* shader;

    unsigned int points_VAO, points_VBO;
    unsigned int camera_VAO, camera_VBO, camera_instance_VBO;

    size_t point_count, pose_count;

    void initCameraGhostGeometry();
    void renderPoints(float base_size, float min_size, float max_size);
    void renderCameraPoses();
};