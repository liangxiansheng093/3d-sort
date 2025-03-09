#pragma once
#include "config.h"

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;

    float mask[32];
    float keypoints[51];  // 17*3 keypoints
};

class TargetBox
{
private:
    float getWidth() { return (x2 - x1); };
    float getHeight() { return (y2 - y1); };

public:
    int x1;
    int y1;
    int x2;
    int y2;

    int cate;
    float score;
};

struct AffineMatrix {
    float value[6];
};

const int bbox_element =
        sizeof(AffineMatrix) / sizeof(float) + 1;  // left, top, right, bottom, confidence, class, keepflag
