#include "splat_parser.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

// A simple utility to read the PLY header and binary data
std::vector<Splat> readGaussianSplats(const std::string& filepath) {
    std::vector<Splat> splats;
    std::ifstream file(filepath, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open Gaussian Splat file: " << filepath << std::endl;
        return splats;
    }

    std::string line;
    int vertexCount = 0;
    
    // Parse PLY header
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            std::stringstream ss(line);
            std::string temp1, temp2;
            ss >> temp1 >> temp2 >> vertexCount;
        }
        if (line.find("end_header") != std::string::npos) {
            break;
        }
    }

    if (vertexCount == 0) {
        std::cerr << "No vertices found in PLY header or invalid file format." << std::endl;
        return splats;
    }

    splats.reserve(vertexCount);
    
    // The standard gaussian splat PLY binary format contains properties in a specific order:
    // x, y, z
    // nx, ny, nz
    // f_dc_0, f_dc_1, f_dc_2
    // f_rest_0 ... f_rest_44 (45 floats for SH bands 1-3)
    // opacity
    // scale_0, scale_1, scale_2
    // rot_0, rot_1, rot_2, rot_3

    // Note: Due to variable SH degrees used in some models, it's safer to read the whole struct
    // if the model format is strictly the standard output. For a complete robust parser, we should
    // parse the property list in the header. For simplicity in this specialized parser, we'll
    // assume the standard output format. Let's compute the byte offset to skip 'nx, ny, nz' and 'f_rest'.
    // If we want this to be robust, we need to read exactly what the file contains based on the header.
    // Let's implement a dynamic parser based on the properties.

    file.seekg(0, std::ios::beg);
    std::vector<std::string> properties;
    while (std::getline(file, line)) {
        if (line.find("property float ") != std::string::npos) {
            properties.push_back(line.substr(15));
            // e.g. "x", "y", "z", "nx", "f_dc_0", "opacity", "scale_0", "rot_0"
        }
        if (line.find("end_header") != std::string::npos) {
            break;
        }
    }

    auto getPropertyIndex = [&](const std::string& name) -> int {
        for (size_t i = 0; i < properties.size(); ++i) {
            if (properties[i].find(name) != std::string::npos || properties[i] == name) return static_cast<int>(i);
        }
        return -1;
    };

    int idxX = getPropertyIndex("x");
    int idxY = getPropertyIndex("y");
    int idxZ = getPropertyIndex("z");
    int idxDC0 = getPropertyIndex("f_dc_0");
    int idxDC1 = getPropertyIndex("f_dc_1");
    int idxDC2 = getPropertyIndex("f_dc_2");
    int idxOpac = getPropertyIndex("opacity");
    int idxScale0 = getPropertyIndex("scale_0");
    int idxScale1 = getPropertyIndex("scale_1");
    int idxScale2 = getPropertyIndex("scale_2");
    int idxRot0 = getPropertyIndex("rot_0"); // scalar part (w)
    int idxRot1 = getPropertyIndex("rot_1"); // x
    int idxRot2 = getPropertyIndex("rot_2"); // y
    int idxRot3 = getPropertyIndex("rot_3"); // z

    size_t propertyCount = properties.size();
    std::vector<float> vertexData(propertyCount);

    // Reading binary vertices
    for (int i = 0; i < vertexCount; ++i) {
        file.read(reinterpret_cast<char*>(vertexData.data()), propertyCount * sizeof(float));

        Splat splat;
        splat.position = glm::vec3(vertexData[idxX], vertexData[idxY], vertexData[idxZ]);
        
        // SH DC component to RGB conversion
        // Standard SH0 = 0.28209479177387814
        // color = SH_DC * SH0 + 0.5
        const float SH_C0 = 0.28209479177387814f;
        splat.color_dc = glm::vec3(
            std::fmax(0.0f, std::fmin(1.0f, vertexData[idxDC0] * SH_C0 + 0.5f)),
            std::fmax(0.0f, std::fmin(1.0f, vertexData[idxDC1] * SH_C0 + 0.5f)),
            std::fmax(0.0f, std::fmin(1.0f, vertexData[idxDC2] * SH_C0 + 0.5f))
        );

        // Opacity is stored as inverse sigmoid
        float opacity = vertexData[idxOpac];
        splat.opacity = 1.0f / (1.0f + std::exp(-opacity));

        // Scale is stored as log, need to exp it
        splat.scale = glm::vec3(
            std::exp(vertexData[idxScale0]),
            std::exp(vertexData[idxScale1]),
            std::exp(vertexData[idxScale2])
        );

        // Rotation is a quaternion, need to normalize
        glm::quat q(vertexData[idxRot0], vertexData[idxRot1], vertexData[idxRot2], vertexData[idxRot3]);
        splat.rotation = glm::normalize(q);
        
        splat.cameraDistance = 0.0f;

        splats.push_back(splat);
    }

    std::cout << "Loaded " << splats.size() << " Gaussian Splats." << std::endl;
    return splats;
}
