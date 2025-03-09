//
// Created by lc on 24-7-19.
//

#ifndef DETECT_ENGINE_YOLOV8_DET_H
#define DETECT_ENGINE_YOLOV8_DET_H

#include <string>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"


// 声明函数
void deserialize_engine(const std::string& engine_name, nvinfer1::IRuntime** runtime,
                        nvinfer1::ICudaEngine** engine, nvinfer1::IExecutionContext** context);
void prepare_buffer(nvinfer1::ICudaEngine* engine, float** device_buffers0, float** device_buffers1,
                    float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device);
void infer(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output,
           int batchsize, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes);
void process_image(cv::Mat img, float* device_buffer[0], cudaStream_t stream, nvinfer1::IExecutionContext* context,
                   float* output_buffer_host, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes,
                   std::vector<std::vector<Detection>>& res_batch);


#endif //DETECT_ENGINE_YOLOV8_DET_H
