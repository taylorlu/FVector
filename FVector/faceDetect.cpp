//
//  faceDetect.cpp
//  FVector
//
//  Created by LuDong on 2018/7/6.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#include <stdio.h>
#include "mtcnn.h"
#include "comm.hpp"

using namespace cv;

cv::Mat drawDetection(const cv::Mat &img, std::vector<Bbox> &box)
{
    cv::Mat show = img.clone();
    int num_box = (int)box.size();
    std::vector<cv::Rect> bbox;
    bbox.resize(num_box);
    for (int i = 0; i < num_box; i++) {
        bbox[i] = cv::Rect(box[i].x1, box[i].y1, box[i].x2 - box[i].x1 + 1, box[i].y2 - box[i].y1 + 1);
        
        for (int j = 0; j < 5; j = j + 1)
        {
            cv::circle(show, cvPoint(box[i].ppoint[j], box[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
        }
    }
    for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
        rectangle(show, (*it), Scalar(0, 0, 255), 2, 8, 0);
    }
    return show;
}

#if 0
int main() {
    
    const char *model_path = "/Users/ludong/Desktop/VideoFace/VideoFace/mtcnn";
    MTCNN mtcnn;
    mtcnn.init(model_path);
    mtcnn.SetMinFace(40);
    
    const char *imgPath1 = "/Users/ludong/Desktop/2.png";
    cv::Mat frame(cvLoadImage(imgPath1));

    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
    std::vector<Bbox> finalBbox;
    clock_t start_time = clock();
    mtcnn.detect(ncnn_img, finalBbox);
    clock_t finish_time = clock();
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "cost " << total_time * 1000 << "ms" << std::endl;

    cv::Mat show = drawDetection(frame, finalBbox);
    imwrite("/Users/ludong/Desktop/zzz.png", show);
    return 0;
}
#endif

