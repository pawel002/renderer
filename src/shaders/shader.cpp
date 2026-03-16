#include "shader.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

Shader::Shader(const char* vert_path, const char* frag_path) {
    string vert_code, frag_code;
    ifstream vert_file, frag_file;

    vert_file.exceptions(ifstream::failbit | ifstream::badbit);
    frag_file.exceptions(ifstream::failbit | ifstream::badbit);

    try {
        vert_file.open(vert_path);
        frag_file.open(frag_path);
        stringstream vert_stream, frag_stream;
        
        vert_stream << vert_file.rdbuf();
        frag_stream << frag_file.rdbuf();
        
        vert_file.close();
        frag_file.close();
        
        vert_code = vert_stream.str();
        frag_code = frag_stream.str();
    }
    catch (ifstream::failure& e) {
        (void)e;
        cerr << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << vert_path << " or " << frag_path << endl;
    }

    const char* vShaderCode = vert_code.c_str();
    const char* fShaderCode = frag_code.c_str();

    GLuint vertex, fragment;

    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");

    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");

    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");

    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void Shader::use() {
    glUseProgram(ID);
}

void Shader::checkCompileErrors(GLuint shader, const string& type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << endl;
        }
    }
}