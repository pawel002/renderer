#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>

#include "camera/camera.h"
#include "point_cloud/renderer.h"
#include "gaussian/renderer.h"

class Application {
public:
    Application();
    ~Application();

    int run();

private:
    GLFWwindow* window;
    Camera camera;

    PointCloudRenderer point_cloud_renderer;
    GaussianRenderer gaussian_renderer;
    
    float delta_time, last_frame, last_x, last_y;
    bool first_mouse, ui_active, show_cameras;

    enum class RenderMode {
        POINT_CLOUD,
        GAUSSIAN_SPLAT
    };

    RenderMode current_mode;

    char points_path[256];
    char images_path[256];
    char splats_path[256];

    float base_point_size, min_point_size, max_point_size;
    size_t point_count, pose_count, splats_count;

    bool initGLFW();
    void initImGui();
    void setUIMode(bool active);

    void processInput();
    void renderUI();
    void cleanup();

    void loadPointsData();
    void loadSplatsData();

    static void mouseCallbackStatic(GLFWwindow* window, double xpos, double ypos);
    static void keyCallbackStatic(GLFWwindow* window, int key, int scancode, int action, int mods);
};
