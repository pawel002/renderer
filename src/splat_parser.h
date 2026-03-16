#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// A structure representing a single 3D Gaussian Splat
struct Splat {
    glm::vec3 position;       // x, y, z
    glm::vec3 scale;          // scale_0, scale_1, scale_2
    glm::quat rotation;       // rot_0, rot_1, rot_2, rot_3 (note: w, x, y, z)
    glm::vec3 color_dc;       // f_dc_0, f_dc_1, f_dc_2 (spherical harmonics DC component)
    float opacity;            // opacities
    
    // For sorting
    float cameraDistance; 
};

// Loads a standard Gaussian Splatting .ply file
std::vector<Splat> readGaussianSplats(const std::string& filepath);
