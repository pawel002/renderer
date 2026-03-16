#include "renderer.h"
#include "cuda_rasterizer.h"
#include <GL/glew.h>
#include <glm/gtc/type_ptr.hpp>

Renderer::Renderer() : shader(nullptr), quadShader(nullptr), cudaRasterizer(new CudaRasterizer()),
                       pointsVAO(0), pointsVBO(0), cameraVAO(0), cameraVBO(0), 
                       quadVAO(0), quadVBO(0), 
                       pointCount(0), splatCount(0) {
}

Renderer::~Renderer() {
    if (shader) {
        delete shader;
    }
    if (pointsVAO) {
        glDeleteVertexArrays(1, &pointsVAO);
        glDeleteBuffers(1, &pointsVBO);
    }
    if (quadShader) {
        delete quadShader;
    }
    if (quadVAO) {
        glDeleteVertexArrays(1, &quadVAO);
        glDeleteBuffers(1, &quadVBO);
    }
    if (cudaRasterizer) {
        delete cudaRasterizer;
    }
}

void Renderer::Init() {
    shader = new Shader("shaders/shader.vert", "shaders/shader.frag");
    // quadShader will render the texture dumped by CUDA
    quadShader = new Shader("shaders/quad.vert", "shaders/quad.frag"); 
    
    cudaRasterizer->Init(1280, 720); // Initial dummy size; will be resized during render if needed
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    InitCameraGhostGeometry();
    InitFullscreenQuad();
}

void Renderer::InitCameraGhostGeometry() {
    // Simple wireframe pyramid representing a camera frustum facing +Z
    float w = 0.1f, h = 0.075f, d = 0.2f; 
    float vertices[] = {
        // Lines from origin to corners
        0.0f, 0.0f, 0.0f,  w,  h, d,
        0.0f, 0.0f, 0.0f, -w,  h, d,
        0.0f, 0.0f, 0.0f,  w, -h, d,
        0.0f, 0.0f, 0.0f, -w, -h, d,
        // Front rectangle
         w,  h, d,  -w,  h, d,
        -w,  h, d,  -w, -h, d,
        -w, -h, d,   w, -h, d,
         w, -h, d,   w,  h, d
    };

    glGenVertexArrays(1, &cameraVAO);
    glGenBuffers(1, &cameraVBO);
    glBindVertexArray(cameraVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cameraVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glDisableVertexAttribArray(1); 
}

void Renderer::InitFullscreenQuad() {
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
}

void Renderer::UpdatePointCloudData(const std::vector<Vertex>& points) {
    if (points.empty()) {
        pointCount = 0;
        return;
    }

    if (pointsVAO == 0) {
        glGenVertexArrays(1, &pointsVAO);
        glGenBuffers(1, &pointsVBO);
    }
    
    glBindVertexArray(pointsVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Vertex), points.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));
    glEnableVertexAttribArray(1);
    
    pointCount = points.size();
}

void Renderer::UpdateSplatData(const std::vector<Splat>& splats) {
    if (splats.empty()) {
        splatCount = 0;
        return;
    }
    splatCount = splats.size();
    
    cudaRasterizer->UpdateSplatData(splats);
}

void Renderer::UpdateCameraData(const std::vector<CameraPose>& cameras) {
    internalCameraPoses = cameras;
}

void Renderer::Render(Camera& camera, float screenWidth, float screenHeight, 
                      float basePointSize, float minPointSize, float maxPointSize, 
                      bool showCameras, int renderMode) {
                      
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 view = camera.GetViewMatrix();
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), screenWidth / screenHeight, 0.1f, 1000.0f);

    if (renderMode == 0) { // POINT CLOUD
        if (!shader) return;

        shader->use();
        glUniformMatrix4fv(glGetUniformLocation(shader->ID, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shader->ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        if (pointCount > 0) {
            glm::mat4 model = glm::mat4(1.0f);
            glUniformMatrix4fv(glGetUniformLocation(shader->ID, "model"), 1, GL_FALSE, glm::value_ptr(model));
            glUniform1i(glGetUniformLocation(shader->ID, "useUniformColor"), 0);
            
            glUniform1f(glGetUniformLocation(shader->ID, "basePointSize"), basePointSize);
            glUniform1f(glGetUniformLocation(shader->ID, "minPointSize"), minPointSize);
            glUniform1f(glGetUniformLocation(shader->ID, "maxPointSize"), maxPointSize);
            
            glBindVertexArray(pointsVAO);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(pointCount));
        }

        // 2. Draw Camera Ghosts (Only in Point Cloud mode for now, or adapt for both later)
        if (showCameras && !internalCameraPoses.empty()) {
            glUniform1i(glGetUniformLocation(shader->ID, "useUniformColor"), 1);
            glUniform3f(glGetUniformLocation(shader->ID, "uniformColor"), 1.0f, 0.0f, 0.0f);
            
            glBindVertexArray(cameraVAO);
            for (const auto& pose : internalCameraPoses) {
                glUniformMatrix4fv(glGetUniformLocation(shader->ID, "model"), 1, GL_FALSE, glm::value_ptr(pose.modelMatrix));
                glDrawArrays(GL_LINES, 0, 16);
            }
        }
    } else if (renderMode == 1) { // GAUSSIAN SPLATTING
        if (!quadShader || splatCount == 0) return;

        // Ensure CUDA rasterizer matches the current viewport size
        int sw = static_cast<int>(screenWidth);
        int sh = static_cast<int>(screenHeight);
        cudaRasterizer->Resize(sw, sh);

        // Execute CUDA rendering pipeline
        cudaRasterizer->Render(view, projection, sw, sh);

        // Draw the output texture to a fullscreen quad
        glDisable(GL_DEPTH_TEST); // We don't need depth testing for a fullscreen quad

        quadShader->use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, cudaRasterizer->GetOutputTexture());
        
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        glEnable(GL_DEPTH_TEST);
    }
}
