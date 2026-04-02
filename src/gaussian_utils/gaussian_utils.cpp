#include "gaussian_utils.h"
#include "../camera/camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

CameraData calculateProjView(const Camera& camera, float fov_x, float fov_y, float znear, float zfar) {
    glm::mat4 flipYZ = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));
    glm::mat4 view = flipYZ * camera.getViewMatrix();

    glm::mat4 proj = glm::mat4(0.0f);
    proj[0][0] = 1.0f / std::tan(fov_x * 0.5f);
    proj[1][1] = 1.0f / std::tan(fov_y * 0.5f);
    proj[2][2] = (zfar + znear) / (zfar - znear);
    proj[2][3] = 1.0f;
    proj[3][2] = -(2.0f * znear * zfar) / (zfar - znear);

    return { view, proj * view, camera.position };
}
