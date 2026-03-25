cmake_minimum_required(VERSION 3.24)
project(Renderer3D LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC_FILES "src/*.cpp" "src/*.cu")

add_executable(Renderer3D main.cpp ${SRC_FILES})

set_target_properties(Renderer3D PROPERTIES CUDA_ARCHITECTURES "native")

target_link_libraries(Renderer3D PRIVATE 
    glfw 
    GLEW::GLEW 
    glm::glm
    OpenGL::GL
    imgui::imgui
    CUDA::cudart
)

target_compile_options(Renderer3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

target_include_directories(Renderer3D PRIVATE ${CUDAToolkit_INCLUDE_DIRS})