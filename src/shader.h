#pragma once
#include <GL/glew.h>
#include <string>

class Shader {
public:
    GLuint ID;
    Shader(const char* vertexPath, const char* fragmentPath);
    void use();

private:
    void checkCompileErrors(GLuint shader, const std::string& type);
};