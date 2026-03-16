#version 330 core
in vec3 vertexColor;
out vec4 outColor;

uniform bool useUniformColor;
uniform vec3 uniformColor;

void main() {
    if (useUniformColor) {
        outColor = vec4(uniformColor, 1.0); 
    } else {
        outColor = vec4(vertexColor, 1.0);
    }
}