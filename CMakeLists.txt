cmake_minimum_required(VERSION 3.15)
project(Renderer3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
# find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC_FILES "src/*.cpp")
# file(GLOB CUDA_SRC_FILES "src/*.cu" "src/cuda_rasterizer/*.cu")

# add_library(CudaRasterizer STATIC ${CUDA_SRC_FILES})
# set_target_properties(CudaRasterizer PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
# target_link_libraries(CudaRasterizer PRIVATE GLEW::GLEW glm::glm)

add_executable(Renderer3D main.cpp ${SRC_FILES})
target_link_libraries(Renderer3D PRIVATE 
    glfw 
    GLEW::GLEW 
    glm::glm
    OpenGL::GL
    imgui::imgui
    # CudaRasterizer
    # CUDA::cudart
)