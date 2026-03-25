#pragma once
#define GLM_ENABLE_EXPERIMENTAL

#include <string>
#include <ostream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

struct Splat {
    glm::vec3 position;
    glm::vec3 scale;
    glm::quat rotation; 
    glm::vec3 color_dc;
    std::vector<float> color_rest;
    float opacity;
};

inline std::ostream& operator<<(std::ostream& os, const Splat& splat) {
    os << "Splat {\n"
       << "  position: " << glm::to_string(splat.position) << "\n"
       << "  scale:    " << glm::to_string(splat.scale) << "\n"
       << "  rotation: " << glm::to_string(splat.rotation) << "\n"
       << "  color_dc: " << glm::to_string(splat.color_dc) << "\n"
       << "  color_rest: [";

    for (size_t i = 0; i < splat.color_rest.size(); ++i) {
        os << splat.color_rest[i];
        if (i != splat.color_rest.size() - 1)
            os << ", ";
    }

    os << "]\n"
       << "  opacity:  " << splat.opacity << "\n"
       << "}";

    return os;
}

struct CameraData {
    glm::mat4 view;
    glm::mat4 proj_view;
    glm::vec3 cam_pos;
};