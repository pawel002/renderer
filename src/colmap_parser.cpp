#include "colmap_parser.h"
#include <fstream>
#include <iostream>

std::vector<Vertex> readPoints3D(const std::string& filepath) {
    std::vector<Vertex> vertices;
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filepath << std::endl;
        return vertices;
    }

    uint64_t num_points;
    file.read(reinterpret_cast<char*>(&num_points), sizeof(uint64_t));
    vertices.reserve(num_points);

    for (uint64_t i = 0; i < num_points; ++i) {
        uint64_t point3D_id;
        double x, y, z, error;
        uint8_t r, g, b;
        uint64_t track_length;

        file.read(reinterpret_cast<char*>(&point3D_id), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&x), sizeof(double));
        file.read(reinterpret_cast<char*>(&y), sizeof(double));
        file.read(reinterpret_cast<char*>(&z), sizeof(double));
        file.read(reinterpret_cast<char*>(&r), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&g), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&b), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&error), sizeof(double));
        file.read(reinterpret_cast<char*>(&track_length), sizeof(uint64_t));

        file.seekg(track_length * (sizeof(uint32_t) + sizeof(uint32_t)), std::ios::cur);

        Vertex v;
        v.position = glm::vec3(x, y, z);
        v.color = glm::vec3(r / 255.0f, g / 255.0f, b / 255.0f);
        vertices.push_back(v);
    }
    return vertices;
}

std::vector<CameraPose> readImages(const std::string& filepath) {
    std::vector<CameraPose> poses;
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filepath << std::endl;
        return poses;
    }

    uint64_t num_reg_images;
    file.read(reinterpret_cast<char*>(&num_reg_images), sizeof(uint64_t));
    poses.reserve(num_reg_images);

    for (uint64_t i = 0; i < num_reg_images; ++i) {
        uint32_t image_id;
        double qw, qx, qy, qz, tx, ty, tz;
        uint32_t camera_id;

        file.read(reinterpret_cast<char*>(&image_id), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&qw), sizeof(double));
        file.read(reinterpret_cast<char*>(&qx), sizeof(double));
        file.read(reinterpret_cast<char*>(&qy), sizeof(double));
        file.read(reinterpret_cast<char*>(&qz), sizeof(double));
        file.read(reinterpret_cast<char*>(&tx), sizeof(double));
        file.read(reinterpret_cast<char*>(&ty), sizeof(double));
        file.read(reinterpret_cast<char*>(&tz), sizeof(double));
        file.read(reinterpret_cast<char*>(&camera_id), sizeof(uint32_t));

        char c;
        do { file.read(&c, 1); } while (c != '\0');

        uint64_t num_points2D;
        file.read(reinterpret_cast<char*>(&num_points2D), sizeof(uint64_t));
        file.seekg(num_points2D * (2 * sizeof(double) + sizeof(uint64_t)), std::ios::cur);

        glm::quat q(qw, qx, qy, qz);
        glm::mat3 R = glm::mat3_cast(q);
        glm::mat3 Rt = glm::transpose(R);
        glm::vec3 T(tx, ty, tz);
        glm::vec3 C = -Rt * T;

        glm::mat4 model(1.0f);
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                model[col][row] = Rt[col][row];
            }
        }
        model[3][0] = C.x;
        model[3][1] = C.y;
        model[3][2] = C.z;

        poses.push_back({model});
    }
    return poses;
}