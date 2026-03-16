#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

struct Point {
    glm::vec3 position;
    glm::vec3 color;
};

struct CameraPose {
    glm::mat4 model_matrix;
};