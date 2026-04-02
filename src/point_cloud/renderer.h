#pragma once

#include <memory>
#include <vector>
#include <glm/glm.hpp>

#include "objects.h"
#include "../shaders/shader.h"

class Camera;

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
    ) const;

    size_t getPointCount() const;
    size_t getPoseCount() const;

private:
    std::unique_ptr<Shader> shader;

    unsigned int points_VAO = 0, points_VBO = 0;
    unsigned int camera_VAO = 0, camera_VBO = 0, camera_instance_VBO = 0;

    size_t point_count = 0, pose_count = 0;

    void initCameraGhostGeometry();
    void renderPoints(float base_size, float min_size, float max_size) const;
    void renderCameraPoses() const;
};