#ifndef SORT_NODE_H
#define SORT_NODE_H

#include "rclcpp/rclcpp.hpp"
#include "lib/matrix/math.hpp"
#include "Sort.h"

#include <thread>
#include <message_filters/synchronizer.h>
#include "visualization_msgs/msg/marker_array.hpp"
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_msg/msg/bounding_boxes.hpp>
#include <cv_msg/msg/objects_states.hpp>
#include <cv_msg/msg/state.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <geometry_msgs/msg/pose_stamped.hpp>


class SortNode : public rclcpp::Node
{
public:
    SortNode();

private:
//    image_transport::ImageTransport it;

    std::shared_ptr<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>> mark_pub;
    std::shared_ptr<rclcpp::Publisher<nav_msgs::msg::Path>> path_pub;
    std::shared_ptr<rclcpp::Publisher<nav_msgs::msg::Path>> ball_vicon_pub;
    std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::PoseStamped>> ball_posestamp_pub;
    std::shared_ptr<rclcpp::Publisher<cv_msg::msg::ObjectsStates>> obj_pub;
    std::shared_ptr<rclcpp::Subscription<nav_msgs::msg::Odometry>> mavros_sub;
    std::shared_ptr<rclcpp::Subscription<cv_msg::msg::BoundingBoxes>> box_sub;

//    image_transport::Subscriber image_sub;
//    image_transport::Publisher image_pub;

    Eigen::Vector3f P;
    Eigen::Matrix3f rotation_matrix;
    double t_now;
    double t_box;
    double d_t_;
    double last_t = 0;

    std::vector<SortRect> rects;
    std::vector<TrackerState> output;

    nav_msgs::msg::Path ballpath;
    nav_msgs::msg::Path ballviconpath;

    void mavrosCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void rectArrayCallback(const cv_msg::msg::BoundingBoxes::SharedPtr msg);
    void TrackingAlgorithm();
    std::vector<Eigen::Vector3f> TrajactoryGenerator(TrackerState state);

    Sort *sort_func;
};

#endif
