//
//  faceEmbedding.cpp
//  FVector
//
//  Created by LuDong on 2018/7/14.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#include <stdio.h>
#include "net.h"
#import <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>
#include "mat.h"

#if 0
using namespace cv;

int main() {
    
    const char *imgPath1 = "/Users/ludong/Desktop/ifset/pic_12.jpg";
    cv::Mat frame = imread(imgPath1);

//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows, 224, 224);
//    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_RGB, frame.cols, frame.rows);
//    ncnn_img = ncnn_img.reshape(160, 160, 3);
    int w = ncnn_img.w;
    int h = ncnn_img.h;
    int c = ncnn_img.c;
    printf("w = %d, h = %d, c = %d\n", w,h,c);
    
    ncnn::Net net;
    int ret = net.load_param("/Users/ludong/Desktop/inception_resnet.param");
    ret = net.load_model("/Users/ludong/Desktop/inception_resnet.bin");
    
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", ncnn_img);
    
    ncnn::Mat out;
    clock_t start_time = clock();
    ex.extract("Flatten", out);
    clock_t finish_time = clock();
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "cost " << total_time * 1000 << "ms" << std::endl;

    return 0;
}
#endif
