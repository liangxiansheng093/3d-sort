#include "rclcpp/rclcpp.hpp"
#include <fstream>
#include <iostream>
#include <thread>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_msg/msg/bounding_box.hpp>
#include <cv_msg/msg/bounding_boxes.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <string>
#include "yolov8_det.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

// using namespace nvinfer1;


class DetectNode : public rclcpp::Node
{
public:
    DetectNode(): Node("detect_node")
    {
        image_sub = this->create_subscription<sensor_msgs::msg::Image>(
                "/D435I/color/image_raw", rclcpp::SensorDataQoS(),
                std::bind(&DetectNode::imageCallback, this, std::placeholders::_1));
        depth_sub = this->create_subscription<sensor_msgs::msg::Image>(
                "/D435I/depth/image_rect_raw", rclcpp::SensorDataQoS(),
                std::bind(&DetectNode::depthCallback, this, std::placeholders::_1));

        boundingboxes_pub = this->create_publisher<cv_msg::msg::BoundingBoxes>("/objects", 10);
        image_frame_pub = this->create_publisher<sensor_msgs::msg::Image>("/image_converter/output_video", 10);

        cudaSetDevice(kGpuId);
        deserialize_engine(engine_name, &runtime, &engine, &context);
        CUDA_CHECK(cudaStreamCreate(&stream));
        cuda_preprocess_init(kMaxInputImageSize);
        auto out_dims = engine->getBindingDimensions(1);
        model_bboxes = out_dims.d[0];

        prepare_buffer(engine, &device_buffers[0], &device_buffers[1],
                       &output_buffer_host, &decode_ptr_host, &decode_ptr_device);

        std::cout << "finish init" << std::endl;
    };

private:

    double colorIntr[4] = {605.95056152, 605.86767578, 317.69219971, 253.31745911};
    double depthIntr[4] = {390.91275024, 390.91275024, 323.81063843, 238.16403198};

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        std::cout << "color image msg get" << std::endl;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        w = cv_ptr->image.cols;
        h = cv_ptr->image.rows;

        process_image(cv_ptr->image, device_buffers, stream, context, output_buffer_host,
                      decode_ptr_host, decode_ptr_device, model_bboxes, res_batch);

        if (!res_batch[0].empty()) {
            need_depth = true;

            for (size_t i = 0; i < res_batch[0].size(); i++)
            {
                res_batch[0][i].bbox[0] = clip(res_batch[0][i].bbox[0], 0, w);
                res_batch[0][i].bbox[1] = clip(res_batch[0][i].bbox[1], 0, h);
                res_batch[0][i].bbox[2] = clip(res_batch[0][i].bbox[2]+1, 0, w);
                res_batch[0][i].bbox[3] = clip(res_batch[0][i].bbox[3]+1, 0, h);
            }
        }

        for (size_t j = 0; j < res_batch.size(); j++) {
            for (size_t k = 0; k < res_batch[j].size(); k++) {
                std::cout << "bbox: [";
                for (size_t b = 0; b < 4; b++) {
                    std::cout << res_batch[j][k].bbox[b];
                    if (b < 3) std::cout << ", ";
                }
                std::cout << "], ";
                std::cout << "conf: " << res_batch[j][k].conf << ", ";
                std::cout << "class_id: " << res_batch[j][k].class_id << std::endl;
            }
        }

        image_frame_pub->publish(*cv_ptr->toImageMsg());
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        std::cout << "depth image msg get" << std::endl;

        if (!need_depth)
        {
            return;
        }

        cv_depth_ptr = cv_bridge::toCvCopy(msg, msg->encoding); //16UC1
        depthImgCopy_ = cv_depth_ptr->image.clone();
        ushort d;

        std::chrono::high_resolution_clock::time_point tic = std::chrono::high_resolution_clock::now();

        depthAlign = cv::Mat::zeros(h, w, depthImgCopy_.type());
        for(int i = 0; i < w; i++)
        {
            for(int j = 0; j < h; j++)
            {
                d = depthImgCopy_.at<ushort>(j,i);
                if(d!=0)
                {
                    float x = (i - depthIntr[2]) * d * 0.001 /depthIntr[0] + 0.0150;
                    float y = (j - depthIntr[3]) * d * 0.001 /depthIntr[1];
                    float z = d * 0.001 - 0.0027;

                    int u = int(x/z * colorIntr[0] + colorIntr[2] + 0.5);
                    int v = int(y/z * colorIntr[1] + colorIntr[3] + 0.5);

                    if(u > 0 && u < w && v > 0 && v < h)
                        depthAlign.at<ushort>(v, u) = d;
                }
            }
        }

        cv_msg::msg::BoundingBoxes bboxes;
        for (size_t i = 0; i < res_batch[0].size(); i++)
        {
            get_depth(res_batch[0][i]);
            if (min_depth == 1e4) {
                continue;
            }

            cv_msg::msg::BoundingBox bbox;
            bbox.probability = res_batch[0][i].conf;
            bbox.xmin = res_batch[0][i].bbox[0];
            bbox.ymin = res_batch[0][i].bbox[1];
            bbox.xmax = res_batch[0][i].bbox[2];
            bbox.ymax = res_batch[0][i].bbox[3];
            bbox.distance = min_depth;
            bbox.id = i;
            bbox.cls = class_names[res_batch[0][i].class_id];

            std::cout << bbox.xmin << " " << bbox.ymin << " " << bbox.xmax << " " << bbox.ymax << std::endl;

            bboxes.bounding_boxes.push_back(bbox);
        }

        bboxes.header = msg->header;
        bboxes.image_header = msg->header;
        boundingboxes_pub->publish(bboxes);

        double compTime = std::chrono::duration_cast<std::chrono::microseconds>
                (std::chrono::high_resolution_clock::now() - tic).count() * 1.0e-3;
        std::cout << "depth time cost (ms): " << compTime <<std::endl;

    }

    int clip (int x, int x_min, int x_max)
    {
        if (x<x_min)
        {x=x_min;}
        else if (x>x_max)
        {x=x_max;}
        return x;
    }

    void get_depth(Detection &res_anchor)
    {
        ushort dp;
        min_depth = 1e4;
        x1 = res_anchor.bbox[0];
        y1 = res_anchor.bbox[1];
        x2 = res_anchor.bbox[2];
        y2 = res_anchor.bbox[3];
        cv::Mat reg = depthAlign(cv::Rect(x1, y1, x2-x1, y2-y1));

        for (int ii = reg.cols/3; ii < reg.cols*2/3; ii++ )
            for (int jj =  reg.rows/3; jj < reg.rows*2/3; jj++) {
                dp = reg.ptr<ushort>(jj)[ii];
                if (dp > 0 && dp < 1e4 && dp < min_depth) {
                    min_depth = dp;
                }
            }
        std::cout << "depth:" << min_depth << std::endl;
    }

private:
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::Image>> image_sub;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::Image>> depth_sub;
    std::shared_ptr<rclcpp::Publisher<cv_msg::msg::BoundingBoxes>> boundingboxes_pub;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> image_frame_pub;

    std::string engine_name = "/home/lc/d435i_ros2/src/detect_engine/weight/yolov8n_uav_ball.engine";
    // Deserialize the engine from file
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    int model_bboxes;
    cudaStream_t stream;

    // Prepare cpu and gpu buffers
    float* device_buffers[2];
    float* output_buffer_host = nullptr;
    float* decode_ptr_host = nullptr;
    float* decode_ptr_device = nullptr;
    std::vector<std::vector<Detection>> res_batch;

    bool need_depth = false;
    double min_depth = 1e4;

    std::vector<std::string> class_names = {"UAV", "ball"};

    cv_bridge::CvImagePtr cv_ptr;
    cv_bridge::CvImagePtr cv_depth_ptr;
    cv::Mat depthImgCopy_;
    cv::Mat depthAlign;
    int w, h, x1, y1, x2, y2;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DetectNode>());
    rclcpp::shutdown();
    return 0;
}
