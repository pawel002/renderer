#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

struct Splat {
    glm::vec3 position;
    glm::vec3 scale;
    glm::quat rotation; 
    glm::vec3 color_dc; 
    float opacity;
};

struct CameraData {
    glm::mat4 view;
    glm::mat4 proj_view;
    glm::vec3 cam_pos;
};