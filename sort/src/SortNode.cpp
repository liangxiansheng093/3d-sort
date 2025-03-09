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
#include <eigen3/Eigen/Geometry>


using matrix::Dcmf;
using matrix::Eulerf;
using matrix::Quatf;
using matrix::Vector3f;


cv_bridge::CvImagePtr cam_image;
cv::Mat ImgCopy_;
cv::Point p3, p4;
cv::Scalar colorRectangle1(0, 0, 255);
rclcpp::Time t_stamp;

class SortNode : public rclcpp::Node
{
public:
    SortNode(): Node("sort_node")
    {
        int maxAge = 2;
        int minHits = 3;
        float iouThreshold = 0.15;
        sort_func = new Sort(maxAge, minHits, iouThreshold);

        box_sub = this->create_subscription<cv_msg::msg::BoundingBoxes>(
                "/objects", rclcpp::SensorDataQoS(),
                std::bind(&SortNode::rectArrayCallback, this, std::placeholders::_1));
        mavros_sub = this->create_subscription<nav_msgs::msg::Odometry>(
                "/uav8/mavros/local_position/odom", rclcpp::SensorDataQoS(),
                std::bind(&SortNode::mavrosCallback, this, std::placeholders::_1));
        image_sub = this->create_subscription<sensor_msgs::msg::Image>(
                "/image_converter/output_video", rclcpp::SensorDataQoS(),
                std::bind(&SortNode::imageCallback, this, std::placeholders::_1));

        mark_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/markers_tracked", 10);
        path_pub = this->create_publisher<nav_msgs::msg::Path>("/ball_path", 10);
        ball_posestamp_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("/ball_pose_tracking", 10);
        obj_pub = this->create_publisher<cv_msg::msg::ObjectsStates>("/objects_states", 10);
        image_pub = this->create_publisher<sensor_msgs::msg::Image>("/tracking_img", 10);

        std::cout << "finish init" << std::endl;
    };

private:
    void mavrosCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::cout << "mavros msg get" << std::endl;
        t_now = rclcpp::Time(msg->header.stamp).seconds();

        float x = msg->pose.pose.orientation.x;
        float y = msg->pose.pose.orientation.y;
        float z = msg->pose.pose.orientation.z;
        float w = msg->pose.pose.orientation.w;

        float x1 = msg->pose.pose.position.y;
        float y1 = -msg->pose.pose.position.x;
        float z1 = msg->pose.pose.position.z;

        float vx = msg->twist.twist.linear.y;
        float vy = -msg->twist.twist.linear.x;
        float vz = msg->twist.twist.linear.z;

        float wx = msg->twist.twist.angular.x;
        float wy = msg->twist.twist.angular.y;
        float wz = msg->twist.twist.angular.z;

        Eigen::Quaternionf Q = Eigen::Quaternionf (w, x, y, z);

        Dcmf R{Quatf(w, x, y, z)};
        const Eulerf euler_angles(R);
        float roll = euler_angles(0);
        float pitch = euler_angles(1);
        float yaw = euler_angles(2) - M_PI/2.f;

        Eigen::Vector3f eulerAngle;
        eulerAngle = Q.matrix().eulerAngles(0,1,2);

        double dt_ = t_now - t_box;

        x1 = x1 - vx * dt_;
        y1 = y1 - vy * dt_;
        z1 = z1 - vz * dt_;
        // uav position
        P << x1, y1, z1;

        // uav orientation
        roll = roll - wx * dt_;
        pitch = pitch - wy * dt_;
        yaw = yaw - wz * dt_;

        // std::cout << "wxyz:" << wx << ", " << wy <<  ", " << wz << std::endl;

        Eigen::AngleAxisf rollAngle(Eigen::AngleAxisf(roll,Eigen::Vector3f::UnitX()));
        Eigen::AngleAxisf pitchAngle(Eigen::AngleAxisf(pitch,Eigen::Vector3f::UnitY()));
        Eigen::AngleAxisf yawAngle(Eigen::AngleAxisf(yaw,Eigen::Vector3f::UnitZ()));

        rotation_matrix = yawAngle * pitchAngle * rollAngle;

        is_mavros_sub = true;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        std::cout << "image msg get" << std::endl;

        if (cam_image)
        {
            std::shared_ptr<sensor_msgs::msg::Image> msg = cam_image->toImageMsg();
            image_pub->publish(*msg);
        }

        cam_image = cv_bridge::toCvCopy(msg, msg->encoding);
        if (cam_image)
        {
            ImgCopy_ = cam_image->image.clone();
        }

        is_image_sub = true;
    }

    void rectArrayCallback(const cv_msg::msg::BoundingBoxes::SharedPtr msg)
    {
        if (!is_mavros_sub || !is_image_sub)
        {
            return;
        }
        std::cout << "rectArray msg get" << std::endl;

        rclcpp::Time start = this->now();
        t_stamp = msg->header.stamp;
        t_box = t_stamp.seconds();

        rects.clear();
        for(auto box : msg->bounding_boxes)
        {
            SortRect rect;
            rect.id = box.id;
            rect.centerX = (box.xmin + box.xmax) / 2;
            rect.centerY = (box.ymin + box.ymax) / 2;
            rect.width = (box.xmax - box.xmin);
            rect.height = (box.ymax - box.ymin);
            rect.distance = (box.distance);
            std::cout << "xmin, ymin, xmax, ymax: " << box.xmin << ", " << box.ymin << ", " << box.xmax << ", " << box.ymax << std::endl;
            rects.push_back(rect);
        }

        std::cout << "rects: " << rects[0].centerX << ", " << rects[0].centerY << ", " << rects[0].width << ", " << rects[0].height << std::endl;

        TrackingAlgorithm();

        rclcpp::Time finish = this->now();
        std::cout << "Tracking time: " << (double)(finish - start).seconds() * 1000 << "ms" << std::endl;
    }

    void TrackingAlgorithm();
    std::vector<Eigen::Vector3f> TrajactoryGenerator(TrackerState state);

private:

    std::shared_ptr<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>> mark_pub;
    std::shared_ptr<rclcpp::Publisher<nav_msgs::msg::Path>> path_pub;
    std::shared_ptr<rclcpp::Publisher<nav_msgs::msg::Path>> ball_vicon_pub;
    std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::PoseStamped>> ball_posestamp_pub;
    std::shared_ptr<rclcpp::Publisher<cv_msg::msg::ObjectsStates>> obj_pub;
    std::shared_ptr<rclcpp::Subscription<nav_msgs::msg::Odometry>> mavros_sub;
    std::shared_ptr<rclcpp::Subscription<cv_msg::msg::BoundingBoxes>> box_sub;

    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::Image>> image_sub;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> image_pub;

    bool is_image_sub = false;
    bool is_mavros_sub = false;

    Eigen::Vector3f P;
    Eigen::Matrix3f rotation_matrix;
    double t_now;
    double t_box;

    std::vector<SortRect> rects;
    std::vector<TrackerState> output;

    nav_msgs::msg::Path ballpath;
    nav_msgs::msg::Path ballviconpath;

    Sort *sort_func;
};


void SortNode::TrackingAlgorithm()
{
    output = sort_func->update(rects, ImgCopy_, rotation_matrix, P);

    int time_point = 1;

    visualization_msgs::msg::MarkerArray markerArrayOutput;
    std::vector<Eigen::Vector3f> predict_points;

    cv_msg::msg::ObjectsStates states_output;
    states_output.header.stamp = t_stamp;
    states_output.header.frame_id = "map";

    ballpath.header.stamp = t_stamp;
    ballpath.header.frame_id = "map";

    for(auto state : output)
    {
        SortRect rect;
        rect.fromTrackerState(state, rotation_matrix, P);
        rect.id = state.id;

        p3.x = ((rect.centerX - rect.width / 2) < 1) ? 1 : (rect.centerX - rect.width / 2);
        p3.y = ((rect.centerY - rect.height / 2) < 1) ? 1 : (rect.centerY - rect.height / 2);
        p4.x = ((rect.centerX + rect.width / 2) > ImgCopy_.cols) ? (ImgCopy_.cols - 1) : (rect.centerX + rect.width / 2);
        p4.y = ((rect.centerY + rect.height / 2) > ImgCopy_.rows) ? (ImgCopy_.rows - 1) : (rect.centerY + rect.height / 2);

        cv::rectangle(cam_image->image, p3, p4, colorRectangle1, 3);
        cv::putText(cam_image->image, std::to_string(rect.id), p3, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        std::cout << "ball id :" << rect.id << std::endl;
        std::cout << "position x :" << state.position(0)/1000 << std::endl;
        std::cout << "position y :" << state.position(1)/1000 << std::endl;
        std::cout << "position z :" << state.position(2)/1000 << std::endl;
        std::cout << "velocity x :" << state.velocity(0)/1000 << std::endl;
        std::cout << "velocity y :" << state.velocity(1)/1000 << std::endl;
        std::cout << "velocity z :" << state.velocity(2)/1000 << std::endl;
        std::cout << "acceleration x :" << state.acceleration(0)/1000 << std::endl;
        std::cout << "acceleration y :" << state.acceleration(1)/1000 << std::endl;
        std::cout << "acceleration z :" << state.acceleration(2)/1000 << std::endl;

        cv_msg::msg::State state_msg;
        state_msg.position.x = state.position(0)/1000;
        state_msg.position.y = state.position(1)/1000;
        state_msg.position.z = state.position(2)/1000;
        state_msg.velocity.x = state.velocity(0)/1000;
        state_msg.velocity.y = state.velocity(1)/1000;
        state_msg.velocity.z = state.velocity(2)/1000;
        state_msg.acceleration.x = state.acceleration(0)/1000;
        state_msg.acceleration.y = state.acceleration(1)/1000;
        state_msg.acceleration.z = state.acceleration(2)/1000;
        states_output.states.push_back(state_msg);
        if (states_output.states.size() > 10)
        {
            states_output.states.erase(states_output.states.begin());
        }

        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = "camera_link";
        marker.frame_locked = true;
        marker.lifetime = rclcpp::Duration(0);
        marker.ns = "bounding_box";
        marker.id = time_point;
        time_point += 1;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.type = visualization_msgs::msg::Marker::SPHERE;

        marker.pose.position.x = state.position(0)/1000;
        marker.pose.position.y = state.position(1)/1000;
        marker.pose.position.z = state.position(2)/1000;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.a = 0.3;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        markerArrayOutput.markers.push_back(marker);
        if (markerArrayOutput.markers.size() > 10)
        {
            markerArrayOutput.markers.erase(markerArrayOutput.markers.begin());
        }

        geometry_msgs::msg::PoseStamped this_pose_stamped;
        this_pose_stamped.header.stamp = t_stamp;
        this_pose_stamped.header.frame_id = "map";
        this_pose_stamped.pose.position.x = state.position(0)/1000;
        this_pose_stamped.pose.position.y = state.position(1)/1000;
        this_pose_stamped.pose.position.z = state.position(2)/1000;
        this_pose_stamped.pose.orientation.x = 0.0;
        this_pose_stamped.pose.orientation.y = 0.0;
        this_pose_stamped.pose.orientation.z = 0.0;
        this_pose_stamped.pose.orientation.w = 1.0;
        ball_posestamp_pub->publish(this_pose_stamped);
        ballpath.poses.push_back(this_pose_stamped);
        if (ballpath.poses.size() > 10)
        {
            ballpath.poses.erase(ballpath.poses.begin());
        }

        predict_points = TrajactoryGenerator(state);
        for (auto point : predict_points)
        {
            marker.header.stamp = this->now();
            marker.header.frame_id = "map";
            marker.frame_locked = true;
            marker.lifetime = rclcpp::Duration(0);
            marker.ns = "bounding_box";
            marker.id = time_point;
            time_point += 1;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.type = visualization_msgs::msg::Marker::SPHERE;

            marker.pose.position.x = point(0)/1000;
            marker.pose.position.y = point(1)/1000;
            marker.pose.position.z = point(2)/1000;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.color.a = 0.3;
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            markerArrayOutput.markers.push_back(marker);
        }

        is_image_sub = false;
        is_mavros_sub = false;
    }

    mark_pub->publish(markerArrayOutput);
    obj_pub->publish(states_output);
    path_pub->publish(ballpath);
}

std::vector<Eigen::Vector3f> SortNode::TrajactoryGenerator(TrackerState state)
{
    float delta_t = 0.2;
    float predict_x = state.position(0);
    float predict_y = state.position(1);
    float predict_z = state.position(2);

    Eigen::Vector3f prediction;
    std::vector<Eigen::Vector3f> predict_points;

    for(int i =1; i <= 15; ++i){

        predict_x = predict_x + state.velocity(0) * delta_t + 0.5 * state.acceleration(0) * delta_t * delta_t;
        predict_y = predict_y + state.velocity(1) * delta_t + 0.5 * state.acceleration(1) * delta_t * delta_t;
        predict_z = predict_z + state.velocity(2) * delta_t + 0.5 * state.acceleration(2) * delta_t * delta_t;

        prediction << predict_x, predict_y, predict_z;
        predict_points.push_back(prediction);
    }

    return predict_points;
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SortNode>());
    rclcpp::shutdown();
    return 0;
}
