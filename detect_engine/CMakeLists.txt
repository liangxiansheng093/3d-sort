cmake_minimum_required(VERSION 3.5)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
endif()

project(detect_engine)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(cv_msg REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)


# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  # cuda
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)

  # tensorrt
  include_directories(/usr/src/TensorRT-8.6.1.6/include)
  link_directories(/usr/src/TensorRT-8.6.1.6/lib)

endif()


add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

# OpenCV
find_package(OpenCV 4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

## Include directories
include_directories(
        include
        ${EIGEN3_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${rclcpp_INCLUDE_DIRS}
        ${std_msgs_INCLUDE_DIRS}
        ${sensor_msgs_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}
)

# List of all libraries to link
list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        # ${CUDA_LIB_DIR}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        nvinfer
        nvinfer_plugin
        )

add_executable(detect_engine
        src/DetectNode.cpp
        src/block.cpp
        src/calibrator.cpp
        src/model.cpp
        src/postprocess.cpp
        src/postprocess.cu
        src/preprocess.cu
        src/yolov8_det.cpp
        )

ament_target_dependencies(detect_engine
        rclcpp
        std_msgs
        sensor_msgs
        message_filters
        cv_bridge
        cv_msg
        OpenCV
        Eigen3
        )
        
target_link_libraries(detect_engine
        ${OpenCV_LIBS}
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        nvinfer
        nvinfer_plugin
        cudart
        myplugins
        )

## Install targets
install(TARGETS detect_engine
        DESTINATION lib/${PROJECT_NAME}
        )

## Install header files
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE
        )

## Ament package
ament_package()
