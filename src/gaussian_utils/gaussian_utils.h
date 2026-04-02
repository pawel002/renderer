#pragma once

#include "../gaussian/objects.h"

class Camera;

CameraData calculateProjView(const Camera& camera, float fov_x, float fov_y,
                              float znear = 0.01f, float zfar = 100.0f);
