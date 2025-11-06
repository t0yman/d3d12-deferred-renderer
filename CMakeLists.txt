cmake_minimum_required(VERSION 3.20)

project(d3d12-deferred-renderer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)
find_package(tinyobjloader REQUIRED)
find_package(Stb REQUIRED)

add_executable(${PROJECT_NAME} 
    source/main.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    glfw
    tinyobjloader::tinyobjloader
    d3d12.lib
    dxgi.lib
    d3dcompiler.lib
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${Stb_INCLUDE_DIR}
)

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE 
        /W4
    )
endif()