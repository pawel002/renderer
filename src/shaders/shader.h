#pragma once
#include <GL/glew.h>
#include <string>

class Shader {
public:
    GLuint ID;
    Shader(const char* vertex_path, const char* frag_path);
    void use();

private:
    void checkCompileErrors(GLuint shader, const std::string& type);
};