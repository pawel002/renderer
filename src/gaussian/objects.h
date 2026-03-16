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