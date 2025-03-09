cmake_minimum_required(VERSION 3.5)
project(sort)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_BUILD_TYPE "Release")

## Find dependencies using ament_cmake
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(cv_msg REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nav_msgs REQUIRED)

## Include directories
include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(sort_node
        src/SortNode.cpp
        src/Sort.cpp
        src/SortRect.cpp
        src/Hungarian.cpp
        src/TrackerState.cpp
        src/Tracker.cpp
        )

## Specify libraries to link a library or executable target against
ament_target_dependencies(sort_node
        rclcpp
        std_msgs
        message_filters
        image_transport
        visualization_msgs
        nav_msgs
        cv_bridge
        cv_msg
        OpenCV
        Eigen3
        )

target_link_libraries(sort_node
        ${OpenCV_LIBRARIES}
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${visualization_msgs_LIBRARIES}
        ${nav_msgs_LIBRARIES}
        )

## Install targets
install(TARGETS sort_node
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
