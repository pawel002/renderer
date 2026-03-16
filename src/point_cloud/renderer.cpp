#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"
#include "../camera/camera.h"

PointCloudRenderer::PointCloudRenderer() :
    shader(nullptr),
    points_VAO(0), points_VBO(0),
    camera_VAO(0), camera_VBO(0), camera_instance_VBO(0),
    point_count(0), pose_count(0) { }

PointCloudRenderer::~PointCloudRenderer() {
    if (shader) delete shader;

    if (points_VAO) {
        glDeleteVertexArrays(1, &points_VAO);
        glDeleteBuffers(1, &points_VBO);
    }

    if (camera_VAO) {
        glDeleteVertexArrays(1, &camera_VAO);
        glDeleteBuffers(1, &camera_VBO);
    }
}

void PointCloudRenderer::init() {
    shader = new Shader(
        "src/shaders/point-cloud/vertex", 
        "src/shaders/point-cloud/fragment"
    );

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    initCameraGhostGeometry();
}

void PointCloudRenderer::initCameraGhostGeometry() {
    float w = 0.1f, h = 0.075f, d = 0.2f; 
    float vertices[] = {
        0.0f, 0.0f, 0.0f,  w,  h, d,
        0.0f, 0.0f, 0.0f, -w,  h, d,
        0.0f, 0.0f, 0.0f,  w, -h, d,
        0.0f, 0.0f, 0.0f, -w, -h, d,
         w,  h, d,  -w,  h, d,
        -w,  h, d,  -w, -h, d,
        -w, -h, d,   w, -h, d,
         w, -h, d,   w,  h, d
    };

    glGenVertexArrays(1, &camera_VAO);
    glGenBuffers(1, &camera_VBO);
    glGenBuffers(1, &camera_instance_VBO);

    glBindVertexArray(camera_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, camera_VBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(0);
    glDisableVertexAttribArray(1); 
}

void PointCloudRenderer::updatePointCloudData(const std::vector<Point>& points) {
    if (points.empty()) {
        point_count = 0;
        return;
    }

    if (points_VAO == 0) {
        glGenVertexArrays(1, &points_VAO);
        glGenBuffers(1, &points_VBO);
    }

    glBindVertexArray(points_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, points_VBO);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Point), points.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Point), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Point), (void*)offsetof(Point, color));
    glEnableVertexAttribArray(1);
    
    point_count = points.size();
}

void PointCloudRenderer::updateCameraData(const std::vector<CameraPose>& cameras) {
    if (cameras.empty()) return;

    std::vector<glm::mat4> poses;
    poses.reserve(cameras.size());

    for (const auto& pose : cameras) {
        poses.push_back(pose.model_matrix);
    }

    glBindVertexArray(camera_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, camera_instance_VBO);
    glBufferData(GL_ARRAY_BUFFER, poses.size() * sizeof(glm::mat4), poses.data(), GL_STATIC_DRAW);

    size_t vec4Size = sizeof(glm::vec4);
    for (int i = 0; i < 4; i++) {
        glEnableVertexAttribArray(2 + i);
        glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(i * vec4Size));
        glVertexAttribDivisor(2 + i, 1); 
    }
    
    pose_count = cameras.size();
}

void PointCloudRenderer::render(
    Camera& camera,
    float screen_width, float screen_height,
    float base_point_size, float min_point_size, float max_point_size,
    bool show_cameras   
) const {
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), screen_width / screen_height, 0.1f, 1000.0f);

    if (!shader) return;

    shader->use();
    glUniformMatrix4fv(glGetUniformLocation(shader->ID, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shader->ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    if (point_count > 0)
        renderPoints(base_point_size, min_point_size, max_point_size);

    if (show_cameras && pose_count > 0)
        renderCameraPoses();
}

void PointCloudRenderer::renderPoints(float base_size, float min_size, float max_size) const {
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shader->ID, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform1i(glGetUniformLocation(shader->ID, "useUniformColor"), 0);
    glUniform1i(glGetUniformLocation(shader->ID, "isInstanced"), 0);
    
    glUniform1f(glGetUniformLocation(shader->ID, "basePointSize"), base_size);
    glUniform1f(glGetUniformLocation(shader->ID, "minPointSize"), min_size);
    glUniform1f(glGetUniformLocation(shader->ID, "maxPointSize"), max_size);

    glBindVertexArray(points_VAO);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count));
}

void PointCloudRenderer::renderCameraPoses() const {
    glUniform1i(glGetUniformLocation(shader->ID, "useUniformColor"), 1);
    glUniform1i(glGetUniformLocation(shader->ID, "isInstanced"), 1);
    glUniform3f(glGetUniformLocation(shader->ID, "uniformColor"), 1.0f, 0.0f, 0.0f);
    
    glBindVertexArray(camera_VAO);
    glDrawArraysInstanced(GL_LINES, 0, 16, static_cast<GLsizei>(pose_count));
}

size_t PointCloudRenderer::getPointCount() const {
    return point_count;
}

size_t PointCloudRenderer::getPoseCount() const {
    return pose_count;
}