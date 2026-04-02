#define _CRT_SECURE_NO_WARNINGS
#include "application.h"
#include <iostream>
#include <cstring>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "point_cloud/colmap_parser.h"
#include "gaussian/gaussian_parser.h"

const char* APP_NAME = "3D Renderer";

const char* POINTS_PATH          = "data/test2/output2/1/points3D.bin";
const char* IMAGES_PATH          = "data/test2/output2/1/images.bin";
const char* SPLATS_PATH          = "data/gs-test2/point_cloud.ply";
const char* TEMPORAL_STATIC_PATH = "data/gs-test2/point_cloud.ply";
const char* TEMPORAL_DYN_FOLDER  = "data/temporal_dynamic";

const int WIDTH = 1280;
const int HEIGHT = 720;
const float WIDTH_F = static_cast<float>(WIDTH);
const float HEIGHT_F = static_cast<float>(HEIGHT);

const float BASE_POINTS_SIZE = 25.0f;
const float MIN_POINTS_SIZE = 1.0f;
const float MAX_POINTS_SIZE = 20.0f;

Application::Application()
    : window(nullptr),
      camera(glm::vec3(0.0f, 0.0f, 5.0f)),
      point_cloud_renderer(),
      gaussian_renderer(),
      temporal_gaussian_renderer(),
      delta_time(0.0f), last_frame(0.0f), last_x(WIDTH_F / 2.0f), last_y(HEIGHT_F / 2.0f),
      first_mouse(true), ui_active(true), show_cameras(true),
      current_mode(RenderMode::POINT_CLOUD),
      base_point_size(BASE_POINTS_SIZE), min_point_size(MIN_POINTS_SIZE), max_point_size(MAX_POINTS_SIZE),
      splat_scale_modifier(1.0f),
      temporal_scale_modifier(1.0f),
      point_count(0), pose_count(0), splats_count(0) {

    snprintf(points_path,           sizeof(points_path),           "%s", POINTS_PATH);
    snprintf(images_path,           sizeof(images_path),           "%s", IMAGES_PATH);
    snprintf(splats_path,           sizeof(splats_path),           "%s", SPLATS_PATH);
    snprintf(temporal_static_path,  sizeof(temporal_static_path),  "%s", TEMPORAL_STATIC_PATH);
    snprintf(temporal_dynamic_folder, sizeof(temporal_dynamic_folder), "%s", TEMPORAL_DYN_FOLDER);
}

Application::~Application() {
}

int Application::run() {
    if (!initGLFW()) return -1;
    initImGui();
    
    point_cloud_renderer.init();
    gaussian_renderer.init(WIDTH, HEIGHT);
    temporal_gaussian_renderer.init(WIDTH, HEIGHT);

    while (!glfwWindowShouldClose(window)) {
        float current_frame = static_cast<float>(glfwGetTime());
        delta_time = current_frame - last_frame;
        last_frame = current_frame;

        processInput();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float screen_width = static_cast<float>(width > 0 ? width : WIDTH_F);
        float screen_height = static_cast<float>(height > 0 ? height : HEIGHT_F);

        if (current_mode == RenderMode::POINT_CLOUD) {
            point_cloud_renderer.render(
                camera,
                screen_width, screen_height,
                base_point_size, min_point_size, max_point_size,
                show_cameras
            );
        }

        if (current_mode == RenderMode::GAUSSIAN_SPLAT) {
            gaussian_renderer.render(
                camera,
                screen_width, screen_height,
                splat_scale_modifier
            );
        }

        if (current_mode == RenderMode::TEMPORAL_GAUSSIAN_SPLAT) {
            if (temporal_gaussian_renderer.isLoaded() && temporal_auto_play) {
                temporal_frame_accum += delta_time * temporal_playback_fps;
                if (temporal_frame_accum >= 1.0f) {
                    int steps = static_cast<int>(temporal_frame_accum);
                    temporal_frame_accum -= steps;
                    int next = (temporal_gaussian_renderer.getCurrentFrame() + steps)
                               % temporal_gaussian_renderer.getFrameCount();
                    temporal_gaussian_renderer.setFrame(next);
                }
            }
            temporal_gaussian_renderer.render(
                camera,
                screen_width, screen_height,
                temporal_scale_modifier
            );
        }

        renderUI();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanup();
    return 0;
}

bool Application::initGLFW() {
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(WIDTH, HEIGHT, APP_NAME, NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);

    glfwSetWindowUserPointer(window, this);
    glfwSetCursorPosCallback(window, mouseCallbackStatic);
    glfwSetKeyCallback(window, keyCallbackStatic);

    if (glewInit() != GLEW_OK) return false;
    return true;
}

void Application::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    setUIMode(true);
}

void Application::setUIMode(bool active) {
    ui_active = active;
    if (ui_active) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        first_mouse = true;
        return;
    }
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void Application::loadPointsData() {
    point_cloud_renderer.updatePointCloudData(readPoints3D(points_path));
    point_cloud_renderer.updateCameraData(readImages(images_path));

    point_count = point_cloud_renderer.getPointCount();
    pose_count = point_cloud_renderer.getPoseCount();
}

void Application::loadSplatsData() {
    auto splats = readGaussianSplats(splats_path);
    gaussian_renderer.updateSplats(splats);
    splats_count = splats.size();
}

void Application::loadTemporalScene() {
    auto mode = (temporal_storage_mode == 0)
                    ? TemporalGaussianRenderer::StorageMode::GPU
                    : TemporalGaussianRenderer::StorageMode::RAM;
    temporal_gaussian_renderer.loadScene(temporal_static_path, temporal_dynamic_folder, mode);
    temporal_frame_accum = 0.0f;
}

void Application::processInput() {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (ui_active) return;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.processKeyboard(FORWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.processKeyboard(BACKWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.processKeyboard(LEFT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.processKeyboard(RIGHT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camera.processKeyboard(UP, delta_time);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camera.processKeyboard(DOWN, delta_time);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camera.processKeyboard(ROLL_LEFT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camera.processKeyboard(ROLL_RIGHT, delta_time);
}

void Application::renderUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("3D Renderer Controls");

    // Performance info
    ImGui::Text("FPS: %.1f (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
    int fb_w, fb_h;
    glfwGetFramebufferSize(window, &fb_w, &fb_h);
    ImGui::Text("Resolution: %d x %d", fb_w, fb_h);

    ImGui::Separator();

    // Camera section
    glm::vec3 camPos = camera.position;
    ImGui::Text("Camera Pos: X: %.2f  Y: %.2f  Z: %.2f", camPos.x, camPos.y, camPos.z);
    ImGui::SliderFloat("Move Speed", &camera.movement_speed, 0.5f, 50.0f);
    ImGui::SliderFloat("Mouse Sensitivity", &camera.mouse_sensitivity, 0.01f, 0.5f);
    if (ImGui::Button("Reset Camera")) {
        camera.reset();
    }

    ImGui::Separator();

    if (ImGui::TreeNode("Controls")) {
        ImGui::BulletText("TAB - Toggle UI / Fly Mode");
        ImGui::BulletText("W/A/S/D - Move");
        ImGui::BulletText("Space/Shift - Up/Down");
        ImGui::BulletText("Q/E - Roll");
        ImGui::BulletText("Mouse - Look around");
        ImGui::BulletText("ESC - Quit");
        ImGui::TreePop();
    }
    
    if (ImGui::BeginTabBar("RenderModes")) {
        if (ImGui::BeginTabItem("Point Cloud")) {
            current_mode = RenderMode::POINT_CLOUD;
            
            ImGui::InputText("Points Path", points_path, IM_ARRAYSIZE(points_path));
            ImGui::InputText("Images Path", images_path, IM_ARRAYSIZE(images_path));
            
            if (ImGui::Button("Load Point Cloud Data")) {
                loadPointsData();
            }

            ImGui::Text("Points loaded: %zu", point_count);
            ImGui::Text("Cameras loaded: %zu", pose_count);
            
            ImGui::Separator();
            ImGui::Text("Point Cloud Scaling");
            ImGui::SliderFloat("Base Size", &base_point_size, 1.0f, 100.0f);
            ImGui::SliderFloat("Min Size", &min_point_size, 0.1f, 5.0f);
            ImGui::SliderFloat("Max Size", &max_point_size, 5.0f, 50.0f);
            
            ImGui::EndTabItem();
        }
        
        if (ImGui::BeginTabItem("Gaussian Splatting")) {
            current_mode = RenderMode::GAUSSIAN_SPLAT;

            ImGui::InputText("Splats path", splats_path, IM_ARRAYSIZE(splats_path));

            if (ImGui::Button("Load Gaussian Splat Data")) {
                loadSplatsData();
            }

            ImGui::Text("Splats loaded: %zu", splats_count);

            ImGui::Separator();
            ImGui::Text("Gaussian Splat Scaling");
            ImGui::SliderFloat("Scale Modifier", &splat_scale_modifier, 0.5f, 2.0f);

            ImGui::Separator();

            if (ImGui::Button("Render Image to PPM")) {
                gaussian_renderer.save_image("image.ppm");
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Temporal Gaussian")) {
            current_mode = RenderMode::TEMPORAL_GAUSSIAN_SPLAT;

            ImGui::InputText("Static PLY path",     temporal_static_path,    IM_ARRAYSIZE(temporal_static_path));
            ImGui::InputText("Dynamic frames folder", temporal_dynamic_folder, IM_ARRAYSIZE(temporal_dynamic_folder));

            ImGui::Text("Storage mode:");
            ImGui::SameLine();
            ImGui::RadioButton("GPU (fast, high VRAM)", &temporal_storage_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("RAM (lower VRAM)",      &temporal_storage_mode, 1);

            if (ImGui::Button("Load Temporal Scene")) {
                loadTemporalScene();
            }

            if (temporal_gaussian_renderer.isLoaded()) {
                ImGui::Separator();
                ImGui::Text("Static splats:  %zu", temporal_gaussian_renderer.getStaticCount());
                ImGui::Text("Dynamic splats: %zu / frame", temporal_gaussian_renderer.getDynamicCount());
                ImGui::Text("Total splats:   %zu", temporal_gaussian_renderer.getTotalSplatCount());
                ImGui::Text("Frames:         %d",  temporal_gaussian_renderer.getFrameCount());

                ImGui::Separator();
                ImGui::Text("Playback");

                // Frame slider
                int frame = temporal_gaussian_renderer.getCurrentFrame();
                int max_frame = temporal_gaussian_renderer.getFrameCount() - 1;
                if (ImGui::SliderInt("Frame", &frame, 0, max_frame)) {
                    temporal_auto_play = false;
                    temporal_frame_accum = 0.0f;
                    temporal_gaussian_renderer.setFrame(frame);
                }

                // Previous / Next buttons
                ImGui::BeginDisabled(frame <= 0);
                if (ImGui::Button("< Prev")) {
                    temporal_auto_play = false;
                    temporal_gaussian_renderer.setFrame(frame - 1);
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::BeginDisabled(frame >= max_frame);
                if (ImGui::Button("Next >")) {
                    temporal_auto_play = false;
                    temporal_gaussian_renderer.setFrame(frame + 1);
                }
                ImGui::EndDisabled();
                ImGui::SameLine();

                // Play / Pause toggle
                if (temporal_auto_play) {
                    if (ImGui::Button("Pause")) {
                        temporal_auto_play   = false;
                        temporal_frame_accum = 0.0f;
                    }
                } else {
                    if (ImGui::Button(" Play")) {
                        temporal_auto_play   = true;
                        temporal_frame_accum = 0.0f;
                    }
                }

                ImGui::SliderFloat("FPS", &temporal_playback_fps, 1.0f, 120.0f, "%.0f fps");

                ImGui::Separator();
                ImGui::Text("Rendering");
                ImGui::SliderFloat("Scale Modifier##T", &temporal_scale_modifier, 0.5f, 2.0f);

                ImGui::Separator();
                if (ImGui::Button("Save Frame to PPM")) {
                    temporal_gaussian_renderer.save_image("temporal_frame.ppm");
                }
            } else {
                ImGui::Separator();
                ImGui::TextDisabled("No scene loaded.");
            }

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::Separator();
    ImGui::Checkbox("Show Camera Ghosts", &show_cameras);

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::cleanup() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
}

void Application::mouseCallbackStatic(GLFWwindow* window, double xpos_d, double ypos_d) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app->ui_active || ImGui::GetIO().WantCaptureMouse) return;

    float xpos = static_cast<float>(xpos_d);
    float ypos = static_cast<float>(ypos_d);

    if (app->first_mouse) {
        app->last_x = xpos; 
        app->last_y = ypos;
        app->first_mouse = false;
    }

    float xoffset = xpos - app->last_x;
    float yoffset = app->last_y - ypos; 
    app->last_x = xpos; 
    app->last_y = ypos;
    app->camera.processMouseMovement(xoffset, yoffset);
}

void Application::keyCallbackStatic(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        app->setUIMode(!app->ui_active);
    }
}
