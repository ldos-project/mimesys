cmake_minimum_required(VERSION 3.10)
project(orchestrator)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

include_directories(/usr/local/include)

add_executable(orchestrator main.cpp)
target_link_libraries(orchestrator pthread numa onnxruntime)
add_executable(send_msg send_msg.cpp)
add_executable(colorctl colorctl.cpp)
