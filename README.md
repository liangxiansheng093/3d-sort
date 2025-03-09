# 3d-sort
For 3d dynamic target detection and motion tracking includes target detection, 3D tracking program.

## 1. 文件环境与内容
### 1.1 推荐环境:
- Ubuntu 20.04
- [ROS2 (foxy)](https://docs.ros.org/en/foxy/Installation.html)
- [cuda`==`12.2](https://developer.nvidia.com/cuda-12-2-0-download-archive)  [cudnn`==`8.9.6](https://developer.nvidia.com/rdp/cudnn-archive)  [tensorrt`==`8.6.1](https://developer.nvidia.com/tensorrt/download)
- [opencv`==`4.8.0 with CUDA](https://opencv.org/releases/)

### 1.2 文件内容:
- __detect_engine__ (目标检测功能包)
- __sort__ (3D跟踪功能包)
- __cv_msg__ (自定义消息类型)

## 2. 相关依赖库安装
### 2.1 opencv with CUDA
- 默认已安装完成CUDA，cuDNN，并添加至默认路径。
1. 下载`opencv4.8.0`和对应`4.8.0`版本的`opencv_contrib`到本地。
```
https://github.com/opencv/opencv
https://github.com/opencv/opencv_contrib
```
2. 创建build文件夹，并进入build文件夹。
```
mkdir build
cd build
```
3. 使用`cmake`来配置变量构建`opencv`。
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN=6.1 -D WITH_CUBLAS=1 -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.8.0/modules -D BUILD_EXAMPLES=ON ..
```
4. 确认`make`命令正确执行完成以后，开始编译并安装`opencv`。
```
make -j8
sudo make install
```

## 3.engine权重文件生成
### 3.1 tensorRT模型生成
参考内容: [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8)
1. 下载大佬开源的`tensorRT`模型生成及部署程序，并将自己的训练权重移动至该文件夹的`yolov8`文件夹下。
```
https://github.com/wang-xinyu/tensorrtx
cp yolov8n.pt code/tensorrtx/yolov8
```
2. 创建并进入`build`文件夹
```
mkdir build
cd build
```
3. 使用`cmake`配置，并通过`make`编译。
```
cmake ..
make
```
4. 将pt权重文件转换成`wts`格式，并移动到build文件夹。
`wts`权重格式: 一种16进制可解释性文本。
```
python3 gen_wts.py -w yolov8n.pt -o yolov8n.wts -t detect
mv yolov8n.wts build
```
5. 通过终端输入指令生成`yolov8n.engine`文件。
```
sudo ./yolov8_det -s yolov8n.wts yolov8n.engine n
```
6.  执行完成指定命令以后，将`yolov8n.engine`权重文件移动`src/detect_engine/weight`下。
### 3.2 libmyplugins.so软链接
1. 第一次运行前，需要对`libmyplugins.so`产生正确的软链接。
```
sudo ln -s /home/nvidia/d435i_ros2/build/detect_engine/libmyplugins.so /usr/local/lib/libmyplugins.so
sudo ldconfig
```
  不然可能出现如下错误: 
```
error while loading shared libraries: libmyplugins.so: cannot open shared object file: no such file or directory
```

## 4.编译并运行功能包
### 4.1 编译工作空间
1. 将`src`文件放到`d435i_ros2`(你的工作空间)，然后直接编译。
```
source /opt/ros/foxy/setup.bash
colcon build
```
### 4.2 执行功能包
1. 连接`d435i`相机(最好使用`USB3.0`接口)，并启动相机程序。
```
ros2 launch realsense2_camera rs_launch.py 
```
  通过`ros2 topic list`查看相机发布的话题。
```
/camera/color/camera_info
/camera/color/image_raw
/camera/color/metadata
/camera/depth/camera_info
/camera/depth/image_rect_raw
/camera/depth/metadata
/camera/extrinsics/depth_to_color
/tf
/tf_static
```
2. 启动`detect_engine`功能包的检测程序。
```
ros2 run detect_engine detect_engine
```
  检测程序接收到RGB彩色图消息后，进行目标检测；在检测到目标后，根据接收到的深度图消息对检测框进行深度值估计，并将图片和检测信息以ROS2话题形式发布。
```
/objects
/image_converter/output_video
```
3. 启动sort功能包的跟踪程序。
```
ros2 run sort sort_node 
```
  跟踪程序接收到飞机位姿速度消息以及检测信息消息后，进行目标跟踪；实现目标的3D轨迹估计和速度加速度计算，然后发布以下新的ROS2话题。
```
/markers_tracked
/ball_path
/ball_pose_tracking
/objects_states
/tracking_img
```
- 跟踪程序不仅需要检测发布的结果和图片，还需要飞机的位姿和速度消息，即`/uav/mavros/local_position/odom`，如果没有，可以使用`rqt`创建一个虚拟的`nav_msgs::msg::Odometry`话题。

