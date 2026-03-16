#include <iostream>

#include "camera.h"

Camera::Camera(glm::vec3 position) : 
    position(position), front(glm::vec3(0.0f, 0.0f, 1.0f)), 
    movement_speed(5.0f), mouse_sensitivity(0.1f), roll_speed(2.0f) {
    position = position;
    world_up = glm::vec3(0.0f, -1.0f, 0.0f); 
    yaw = 90.0f;
    pitch = 0.0f;
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() {
    return glm::lookAt(position, position + front, up);
}

void Camera::processKeyboard(CameraMovement direction, float delta_time) {
    float velocity = movement_speed * delta_time;
    glm::vec3 flat_front = glm::normalize(front - world_up * glm::dot(front, world_up));

    if (direction == FORWARD)  position += flat_front * velocity;
    if (direction == BACKWARD) position -= flat_front * velocity;
    if (direction == LEFT)     position -= right * velocity;
    if (direction == RIGHT)    position += right * velocity;

    if (direction == UP)       position += world_up * velocity;
    if (direction == DOWN)     position -= world_up * velocity;

    if (direction == ROLL_LEFT || direction == ROLL_RIGHT) {
        float roll_velocity = roll_speed * delta_time;

        if (direction == ROLL_LEFT) roll_velocity = -roll_velocity;

        glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), roll_velocity, front);
        world_up = glm::normalize(glm::vec3(rotationMatrix * glm::vec4(world_up, 0.0f)));
        
        updateCameraVectors();
    }
}

void Camera::processMouseMovement(float xoffset, float yoffset) {
    xoffset *= mouse_sensitivity;
    yoffset *= mouse_sensitivity;

    yaw -= xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)  pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    updateCameraVectors();
}

void Camera::updateCameraVectors() {
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = -sin(glm::radians(pitch)); 
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    front = glm::normalize(front);
    
    right = glm::normalize(glm::cross(front, world_up));
    up    = glm::normalize(glm::cross(right, front));
}
