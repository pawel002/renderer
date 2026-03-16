#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "colmap_parser.h"
#include "splat_parser.h"
#include "shader.h"
#include "camera.h"

class CudaRasterizer;

class Renderer {
public:
    Renderer();
    ~Renderer();

    void Init();
    
    void UpdatePointCloudData(const std::vector<Vertex>& points);
    void UpdateCameraData(const std::vector<CameraPose>& cameras);
    void UpdateSplatData(const std::vector<Splat>& splats);

    void Render(Camera& camera, float screenWidth, float screenHeight, 
                float basePointSize, float minPointSize, float maxPointSize, 
                bool showCameras, int renderMode);

    size_t GetSplatCount() const { return splatCount; }

private:
    Shader* shader;
    Shader* quadShader;
    CudaRasterizer* cudaRasterizer;
    
    unsigned int pointsVAO, pointsVBO;
    unsigned int cameraVAO, cameraVBO;
    
    // Fullscreen quad for drawing CUDA PBO texture
    unsigned int quadVAO, quadVBO;
    
    size_t pointCount;
    size_t splatCount;
    
    std::vector<Splat> internalSplats;
    std::vector<unsigned int> splatIndices;
    std::vector<CameraPose> internalCameraPoses;

    void InitCameraGhostGeometry();
    void InitFullscreenQuad();
};
