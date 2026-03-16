#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>

#include "camera/camera.h"
#include "renderer.h"
#include "colmap_parser.h"

class Application {
public:
    Application();
    ~Application();

    int run();

private:
    GLFWwindow* window;
    Camera camera;
    Renderer renderer;
    
    float deltaTime, lastFrame, lastX, lastY;
    bool firstMouse, uiActive, showCameras;

    enum class RenderMode {
        POINT_CLOUD,
        GAUSSIAN_SPLAT
    };

    RenderMode currentMode;

    char pointsPath[256];
    char imagesPath[256];
    char splatPath[256];

    float basePointSize;
    float minPointSize;
    float maxPointSize;
    
    std::vector<Vertex> pointCloud;
    std::vector<CameraPose> cameraPoses;

    bool initGLFW();
    void initImGui();
    void setUIMode(bool active);
    void loadData();
    void loadSplatData();
    void processInput();
    void renderUI();
    void cleanup();

    static void mouseCallbackStatic(GLFWwindow* window, double xpos, double ypos);
    static void keyCallbackStatic(GLFWwindow* window, int key, int scancode, int action, int mods);
};
