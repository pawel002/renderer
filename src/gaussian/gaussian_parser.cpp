#include <fstream>
#include <iostream>
#include <sstream>

#include "gaussian_parser.h"

const float SH_C0 = 0.28209479177387814f;

std::vector<Splat> readGaussianSplats(const std::string& file_path) {
    std::vector<Splat> splats;
    std::ifstream file(file_path, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open Gaussian Splat file: " << file_path << std::endl;
        return splats;
    }

    std::string line;
    int vertex_count = 0;
    
    while (std::getline(file, line)) {

        if (line.find("element vertex") != std::string::npos) {
            std::stringstream ss(line);
            std::string temp1, temp2;
            ss >> temp1 >> temp2 >> vertex_count;
        }

        if (line.find("end_header") != std::string::npos) {
            break;
        }
    }

    if (vertex_count == 0) {
        std::cerr << "No vertices found in PLY header or invalid file format." << std::endl;
        return splats;
    }

    splats.reserve(vertex_count);

    file.seekg(0, std::ios::beg);
    std::vector<std::string> properties;
    while (std::getline(file, line)) {
        if (line.find("property float ") != std::string::npos) {
            properties.push_back(line.substr(15));
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

    int idxRot0 = getPropertyIndex("rot_0");
    int idxRot1 = getPropertyIndex("rot_1");
    int idxRot2 = getPropertyIndex("rot_2");
    int idxRot3 = getPropertyIndex("rot_3");

    size_t propertyCount = properties.size();
    std::vector<float> vertexData(propertyCount);

    for (int i = 0; i < vertex_count; ++i) {
        file.read(reinterpret_cast<char*>(vertexData.data()), propertyCount * sizeof(float));

        Splat splat;
        splat.position = glm::vec3(vertexData[idxX], vertexData[idxY], vertexData[idxZ]);
        
        splat.color_dc = glm::vec3(
            std::fmax(0.0f, std::fmin(1.0f, vertexData[idxDC0] * SH_C0 + 0.5f)),
            std::fmax(0.0f, std::fmin(1.0f, vertexData[idxDC1] * SH_C0 + 0.5f)),
            std::fmax(0.0f, std::fmin(1.0f, vertexData[idxDC2] * SH_C0 + 0.5f))
        );

        float opacity = vertexData[idxOpac];
        splat.opacity = 1.0f / (1.0f + std::exp(-opacity));

        splat.scale = glm::vec3(
            std::exp(vertexData[idxScale0]),
            std::exp(vertexData[idxScale1]),
            std::exp(vertexData[idxScale2])
        );

        glm::quat q(vertexData[idxRot0], vertexData[idxRot1], vertexData[idxRot2], vertexData[idxRot3]);
        splat.rotation = glm::normalize(q);

        splats.push_back(splat);
    }

    return splats;
}