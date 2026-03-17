#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D renderTex;

void main() {
    vec2 uv = vec2(TexCoords.x, 1.0 - TexCoords.y); 

    float r = texture(renderTex, vec2(uv.x, uv.y / 3.0)).r;
    float g = texture(renderTex, vec2(uv.x, uv.y / 3.0 + 1.0/3.0)).r;
    float b = texture(renderTex, vec2(uv.x, uv.y / 3.0 + 2.0/3.0)).r;

    FragColor = vec4(r, g, b, 1.0);
}