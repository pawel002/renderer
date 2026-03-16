#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in mat4 instanceMatrix; 

out vec3 vertexColor;

uniform mat4 model;
uniform bool isInstanced; 

uniform mat4 view;
uniform mat4 projection;

uniform float basePointSize;
uniform float minPointSize;
uniform float maxPointSize;

void main() {
    vertexColor = aColor;
    
    mat4 currentModel = isInstanced ? instanceMatrix : model;
    vec4 viewPos = view * currentModel * vec4(aPos, 1.0);
    gl_Position = projection * viewPos;
    
    float distance = length(viewPos.xyz);
    
    float size = basePointSize / distance;
    gl_PointSize = clamp(size, minPointSize, maxPointSize);
}