#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "gaussian_parser.h"

const float SH_C0 = 0.28209479177387814f;

std::vector<Splat> readGaussianSplats(const std::string& file_path) {
    std::vector<Splat> splats;
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open Gaussian Splat file: " << file_path << std::endl;
        return splats;
    }

    // Single-pass header parsing: extract vertex count and property indices
    std::string line;
    int vertex_count = 0;
    std::unordered_map<std::string, int> property_map;
    int property_idx = 0;

    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            std::stringstream ss(line);
            std::string temp1, temp2;
            ss >> temp1 >> temp2 >> vertex_count;
        }

        if (line.find("property float ") != std::string::npos) {
            std::string name = line.substr(15);
            property_map[name] = property_idx++;
        }

        if (line.find("end_header") != std::string::npos) {
            break;
        }
    }

    if (vertex_count == 0) {
        std::cerr << "No vertices found in PLY header or invalid file format." << std::endl;
        return splats;
    }

    // Lookup property indices with validation
    auto getProp = [&](const std::string& name) -> int {
        auto it = property_map.find(name);
        return (it != property_map.end()) ? it->second : -1;
    };

    int idxX = getProp("x"), idxY = getProp("y"), idxZ = getProp("z");
    int idxDC0 = getProp("f_dc_0"), idxDC1 = getProp("f_dc_1"), idxDC2 = getProp("f_dc_2");
    int idxOpac = getProp("opacity");
    int idxScale0 = getProp("scale_0"), idxScale1 = getProp("scale_1"), idxScale2 = getProp("scale_2");
    int idxRot0 = getProp("rot_0"), idxRot1 = getProp("rot_1"), idxRot2 = getProp("rot_2"), idxRot3 = getProp("rot_3");

    if (idxX < 0 || idxY < 0 || idxZ < 0 ||
        idxDC0 < 0 || idxDC1 < 0 || idxDC2 < 0 ||
        idxOpac < 0 ||
        idxScale0 < 0 || idxScale1 < 0 || idxScale2 < 0 ||
        idxRot0 < 0 || idxRot1 < 0 || idxRot2 < 0 || idxRot3 < 0) {
        std::cerr << "PLY file is missing required properties: " << file_path << std::endl;
        return splats;
    }

    size_t propertyCount = property_map.size();
    std::vector<float> vertexData(propertyCount);
    size_t expected_bytes = propertyCount * sizeof(float);

    splats.reserve(vertex_count);

    for (int i = 0; i < vertex_count; ++i) {
        file.read(reinterpret_cast<char*>(vertexData.data()), expected_bytes);
        if (file.gcount() != static_cast<std::streamsize>(expected_bytes)) {
            std::cerr << "PLY file truncated at vertex " << i << " of " << vertex_count << std::endl;
            break;
        }

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
        float len = glm::length(q);
        splat.rotation = (len > 1e-8f) ? q / len : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

        splats.push_back(splat);
    }

    return splats;
}
