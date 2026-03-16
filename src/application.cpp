#define _CRT_SECURE_NO_WARNINGS
#include "application.h"
#include <iostream>
#include <cstring>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "splat_parser.h"

using namespace std;

const char* POINTS_PATH = "data/test2/output2/1/points3D.bin\0";
const char* IMAGES_PATH = "data/test2/output2/1/images.bin\0";
const char* SPLAT_PATH = "data/gs-test1/point_cloud.ply\0";

Application::Application() 
    : window(nullptr),
      camera(glm::vec3(0.0f, 0.0f, 5.0f)), deltaTime(0.0f), lastFrame(0.0f), 
      lastX(1280.0f / 2.0f), lastY(720.0f / 2.0f), firstMouse(true), showCameras(true),
      uiActive(true), currentMode(RenderMode::POINT_CLOUD), basePointSize(25.0f), minPointSize(1.0f), maxPointSize(20.0f) {
    
    strncpy(pointsPath, POINTS_PATH, sizeof(POINTS_PATH));
    strncpy(imagesPath, IMAGES_PATH, sizeof(IMAGES_PATH));
    strncpy(splatPath, SPLAT_PATH, sizeof(SPLAT_PATH));
}

Application::~Application() {
}

int Application::Run() {
    if (!InitGLFW()) return -1;
    InitImGui();
    
    renderer.Init();

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        ProcessInput();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float screenWidth = static_cast<float>(width > 0 ? width : 1280);
        float screenHeight = static_cast<float>(height > 0 ? height : 720);

        renderer.Render(camera, screenWidth, screenHeight, basePointSize, minPointSize, maxPointSize, showCameras, static_cast<int>(currentMode));

        RenderUI();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    Cleanup();
    return 0;
}

bool Application::InitGLFW() {
    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1280, 720, "COLMAP Renderer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);

    glfwSetWindowUserPointer(window, this);
    glfwSetCursorPosCallback(window, MouseCallbackStatic);
    glfwSetKeyCallback(window, KeyCallbackStatic);

    if (glewInit() != GLEW_OK) return false;
    return true;
}

void Application::InitImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    SetUIMode(true); // Start with UI active
}

void Application::SetUIMode(bool active) {
    uiActive = active;
    if (uiActive) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        firstMouse = true;
    } else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}

void Application::LoadData() {
    // Load Points
    pointCloud = readPoints3D(pointsPath);
    renderer.UpdatePointCloudData(pointCloud);

    // Load Cameras
    cameraPoses = readImages(imagesPath);
    renderer.UpdateCameraData(cameraPoses);
}

void Application::LoadSplatData() {
    std::vector<Splat> splats = readGaussianSplats(splatPath);
    renderer.UpdateSplatData(splats);
}

void Application::ProcessInput() {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (uiActive) return; // Don't move camera if typing in UI

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camera.ProcessKeyboard(DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camera.ProcessKeyboard(ROLL_LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camera.ProcessKeyboard(ROLL_RIGHT, deltaTime);
}

void Application::RenderUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("COLMAP & 3DGS Controls");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Separator();
    ImGui::Text("Press TAB to toggle UI / Camera Fly Mode");
    ImGui::Separator();
    
    if (ImGui::BeginTabBar("RenderModes")) {
        if (ImGui::BeginTabItem("Point Cloud")) {
            currentMode = RenderMode::POINT_CLOUD;
            
            ImGui::InputText("Points Path", pointsPath, IM_ARRAYSIZE(pointsPath));
            ImGui::InputText("Images Path", imagesPath, IM_ARRAYSIZE(imagesPath));
            
            if (ImGui::Button("Load Point Cloud Data")) {
                LoadData();
            }

            ImGui::Text("Points loaded: %zu", pointCloud.size());
            ImGui::Text("Cameras loaded: %zu", cameraPoses.size());
            
            ImGui::Separator();
            ImGui::Text("Point Cloud Scaling");
            ImGui::SliderFloat("Base Size", &basePointSize, 1.0f, 100.0f);
            ImGui::SliderFloat("Min Size", &minPointSize, 0.1f, 5.0f);
            ImGui::SliderFloat("Max Size", &maxPointSize, 5.0f, 50.0f);
            
            ImGui::EndTabItem();
        }
        
        if (ImGui::BeginTabItem("Gaussian Splatting")) {
            currentMode = RenderMode::GAUSSIAN_SPLAT;
            
            ImGui::InputText("Splat PLY Path", splatPath, IM_ARRAYSIZE(splatPath));
            
            if (ImGui::Button("Load Gaussian Splats")) {
                LoadSplatData();
            }
            
            ImGui::Text("Splats loaded: %zu", renderer.GetSplatCount());
            
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();
    ImGui::Checkbox("Show Camera Ghosts", &showCameras);

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::Cleanup() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
}

void Application::MouseCallbackStatic(GLFWwindow* window, double xpos_d, double ypos_d) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app->uiActive || ImGui::GetIO().WantCaptureMouse) return;

    float xpos = static_cast<float>(xpos_d);
    float ypos = static_cast<float>(ypos_d);
    
    if (app->firstMouse) {
        app->lastX = xpos; 
        app->lastY = ypos;
        app->firstMouse = false;
    }

    float xoffset = xpos - app->lastX;
    float yoffset = app->lastY - ypos; 
    app->lastX = xpos; 
    app->lastY = ypos;
    app->camera.ProcessMouseMovement(xoffset, yoffset);
}

void Application::KeyCallbackStatic(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        app->SetUIMode(!app->uiActive);
    }
}
