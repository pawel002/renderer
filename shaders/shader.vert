#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float basePointSize;
uniform float minPointSize;
uniform float maxPointSize;

void main() {
    vertexColor = aColor;
    
    // 1. Calculate the position relative to the camera
    vec4 viewPos = view * model * vec4(aPos, 1.0);
    gl_Position = projection * viewPos;
    
    // 2. Calculate distance to the camera
    float distance = length(viewPos.xyz);
    
    // 3. Scale the point size inversely proportional to the distance
    // (You might need to multiply by a constant depending on your scene scale)
    float size = basePointSize / distance;
    
    // 4. Clamp the size so points don't get infinitely large or completely disappear
    gl_PointSize = clamp(size, minPointSize, maxPointSize);
}