# 3D renderer

Lightweight Gaussian Splatting rendering application written in C++ using OpenGL and CUDA. Compiled using Cmake.

TODO:

- ~~write a bridge between my splats and app data to the cuda rasterization module.~~

- ~~camera is upside down in Gaussian Renderer.~~

- ~~fix actual gaussian splats (right now it only shows points). investigate more: https://gitlab.inria.fr/sibr/sibr_core/-/blob/gaussian_code_release/src/projects/gaussianviewer/apps/gaussianViewer/main.cpp~~

- theoretical speed ups (for now current preformance is more than enough)

1. half precision - cut the memory and computation cost by half with gaussians

2. Asynchronous CUDA Streams: Your pipeline currently runs entirely on the default CUDA stream (0). This forces the CPU and GPU to execute sequentially. If you create a custom cudaStream_t, you can overlap the CPU's preparation of frame $N+1$ (ImGui, camera math, etc.) while the GPU is still rasterizing frame $N$.

3. Rewriting entire renderer to use dynamic gaussian sorting instead of batching and tiling (PROBABLY NOT POSSIBLE)
