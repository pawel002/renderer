#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum CameraMovement { FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN, ROLL_LEFT, ROLL_RIGHT };

class Camera {
public:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 world_up;

    float yaw;
    float pitch;
    float movement_speed;
    float mouse_sensitivity;
    float roll_speed;

    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, -5.0f));

    glm::mat4 getViewMatrix() const;
    void processKeyboard(CameraMovement direction, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset);
    void reset(glm::vec3 pos = glm::vec3(0.0f, 0.0f, 5.0f));

private:
    void updateCameraVectors();
};