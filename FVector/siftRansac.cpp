//
//  siftRansac.cpp
//  FVector
//
//  Created by LuDong on 2018/2/25.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#include "siftRansac.hpp"
using namespace std;
using namespace cv;
#if 1
#include <sys/time.h>

double difftimeval(const struct timeval *start, const struct timeval *end)
{
    double d;
    time_t s;
    suseconds_t u;
    
    s = start->tv_sec - end->tv_sec;
    u = start->tv_usec - end->tv_usec;
    //if (u < 0)
    //        --s;
    
    d = s;
    d *= 1000000.0;//1 秒 = 10^6 微秒
    d += u;
    
    return d;
}


void matchTwoImage() {
    
    const char *imgPath1 = "/Users/ludong/Desktop/detect.png";
    const char *imgPath2 = "/Users/ludong/Desktop/scene.png";
    Mat imgMat(cvLoadImage(imgPath1));
    Mat imgMat2(cvLoadImage(imgPath2));
    
    BFMatcher matcher;
    vector<DMatch> matches;
    
    SiftDescriptorExtractor detector;
    vector<KeyPoint> stKeypoints;
    detector.detect(imgMat, stKeypoints);  //standard == train
    Mat stDescriptors;
    detector.compute(imgMat, stKeypoints, stDescriptors);
    
    printf("%ld\n", stDescriptors.dataend-stDescriptors.datastart);
    
    vector<KeyPoint> qKpoints;
    detector.detect(imgMat2, qKpoints);    //other == query
    Mat qDescs;
    detector.compute(imgMat2, qKpoints, qDescs);
    
    matcher.match(qDescs, stDescriptors, matches);  //brute match
    
    vector<KeyPoint> R_keypoint01,R_keypoint02;
    for (size_t i=0;i<matches.size();i++) {
        R_keypoint01.push_back(stKeypoints[matches[i].trainIdx]);
        R_keypoint02.push_back(qKpoints[matches[i].queryIdx]);
    }
    
    vector<Point2f>p01,p02;
    for (size_t i=0;i<matches.size();i++) {
        p01.push_back(R_keypoint01[i].pt);
        p02.push_back(R_keypoint02[i].pt);
    }
    
    vector<uchar> RansacStatus;
//    Mat Fundamental = findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);
    Mat Fundamental = findHomography(p01, p02, CV_RANSAC);
    
//    vector<KeyPoint> RR_keypoint01,RR_keypoint02;
//    vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
//    int index=0;
//    for (size_t i=0;i<matches.size();i++) {
//        if (RansacStatus[i]!=0) {
//            index++;
//        }
//    }
//    printf("RANSAC Count = %d\n", index);
    
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( imgMat2.cols, 0 );
    obj_corners[2] = cvPoint( imgMat2.cols, imgMat2.rows );
    obj_corners[3] = cvPoint( 0, imgMat2.rows );
    std::vector<Point2f> scene_corners(4);
    
    perspectiveTransform( obj_corners, scene_corners, Fundamental);
    
    circle(imgMat2, scene_corners[0], 3, Scalar(255,0,0));
    circle(imgMat2, scene_corners[1], 3, Scalar(255,0,0));
    circle(imgMat2, scene_corners[2], 3, Scalar(255,0,0));
    circle(imgMat2, scene_corners[3], 3, Scalar(255,0,0));
    
    imwrite("/Users/ludong/Desktop/scene2.png", imgMat2);
    printf("====\n");
}

Mat matchMaxRANSAC(Mat standardMat, vector<Mat> queryVector) {
    
    BFMatcher matcher;
    vector<DMatch> matches;

    SiftDescriptorExtractor detector;
    vector<KeyPoint> stKeypoints;
    detector.detect(standardMat, stKeypoints);  //standard == train
    Mat stDescriptors;
    detector.compute(standardMat, stKeypoints, stDescriptors);
    
    Mat ransacDescriptor;
    int maxMatchCount = 0;
    for(int i=0; i<queryVector.size(); i++) {
        
        Mat qMat = queryVector[i];
        vector<KeyPoint> qKpoints;
        detector.detect(qMat, qKpoints);    //other == query
        Mat qDescs;
        detector.compute(qMat, qKpoints, qDescs);
        
        matcher.match(qDescs, stDescriptors, matches);  //brute match
        
        vector<Point2f> p01,p02;
        for (size_t i=0;i<matches.size();i++) {
            p01.push_back(stKeypoints[matches[i].trainIdx].pt);
            p02.push_back(qKpoints[matches[i].queryIdx].pt);
        }
        
        vector<uchar> RansacStatus;
        Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);
        
        vector<KeyPoint> RR_keypoint01,RR_keypoint02;
        vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
        int index=0;
        for (size_t i=0;i<matches.size();i++) {
            if (RansacStatus[i]!=0) {
                index++;
            }
        }
        if(maxMatchCount<index) {   //Do copy for store, We get the MAX Match of RANSAC
            maxMatchCount = index;
            ransacDescriptor.release();
            for (size_t i=0;i<matches.size();i++) {
                if (RansacStatus[i]!=0) {
                    ransacDescriptor.push_back(stDescriptors.row(matches[i].trainIdx));
                }
            }
        }
    }
    return ransacDescriptor;
}



int main() {
    
//    const char *rootDir = "/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples5";
//    std::vector<std::string> folders;
//    readDirectory(rootDir, folders, 1);
//
//    for(int i=0; i<folders.size(); i++) {
//
//        printf("%s\n", folders[i].c_str());
//        std::vector<std::string> files;
//        readDirectory(folders[i].c_str(), files, 0);
//
//        vector<Mat> queryVector;
//        for(int j=0;j<files.size();j++) {
//            IplImage *image = cvLoadImage(files[j].c_str());
//            if(image==NULL)
//                continue;
//            Mat imgMat(image);
//            cvtColor(imgMat, imgMat, CV_BGR2GRAY);
//            queryVector.push_back(imgMat);
//        }
//        String standardDir = "/Users/ludong/Desktop/frontModel/";
//        char name[100] = {0};
//        sprintf(name, "standard%02d.jpg", i);
//        IplImage *sImage = cvLoadImage(standardDir.append(name).c_str());
//        Mat sMat(sImage);
//        cvtColor(sMat, sMat, CV_BGR2GRAY);
//        Mat ransacDescriptor = matchMaxRANSAC(sMat, queryVector);
//        saveMatrix((float *)ransacDescriptor.data, ransacDescriptor.rows, SIFT_DIMENSION, standardDir.substr(0, standardDir.size()-4).c_str());
//    }
    
    struct timeval tv1, tv2;
    gettimeofday(&tv1,NULL);
    matchTwoImage();
    gettimeofday(&tv2,NULL);
    printf("===%.0f\n", difftimeval(&tv2, &tv1));
    return 0;
}
#endif


