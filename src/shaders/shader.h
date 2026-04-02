#pragma once
#include <GL/glew.h>
#include <string>
#include <unordered_map>

class Shader {
public:
    GLuint ID;
    Shader(const char* vertex_path, const char* frag_path);
    ~Shader();
    void use();
    GLint getUniform(const std::string& name);

private:
    std::unordered_map<std::string, GLint> uniform_cache;
    void checkCompileErrors(GLuint shader, const std::string& type);
};