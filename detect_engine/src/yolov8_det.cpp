#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"
#include "yolov8_det.h"


Logger gLogger;
// using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void deserialize_engine(const std::string& engine_name, nvinfer1::IRuntime** runtime, nvinfer1::ICudaEngine** engine,
                        nvinfer1::IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = nvinfer1::createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void prepare_buffer(nvinfer1::ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                    float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    // Allocate memory for decode_ptr_host and copy to device
    *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
    CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
}

void infer(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize,
           float* decode_ptr_host, float* decode_ptr_device, int model_bboxes) {
    // infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    context.enqueue(batchsize, buffers, stream, nullptr);
    CUDA_CHECK(
            cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
    cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
    cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);  //cuda nms
    CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                               sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                               stream));
    auto end = std::chrono::system_clock::now();
    std::cout << "inference and gpu postprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void process_image(cv::Mat img, float* device_buffers[0], cudaStream_t stream, nvinfer1::IExecutionContext* context,
                   float* output_buffer_host, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes,
                   std::vector<std::vector<Detection>>& res_batch) {
    res_batch.clear();
    // Get a batch of images
    std::vector<cv::Mat> img_batch;
    img_batch.push_back(img);

    // Preprocess
    cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
    // Run inference
    infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host,
          decode_ptr_device, model_bboxes);
    //Process gpu decode and nms results
    batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);

}
