//
//  comm.hpp
//  FVector
//
//  Created by LuDong on 2018/1/26.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#ifndef comm_hpp
#define comm_hpp

#include <stdio.h>
#include "generic.h"
#include "mathop.h"
#include "random.h"
#include "gmm.h"
#include "fisher.h"
#include "sift.h"
#include "dsift.h"
#include "mser.h"
#include "AKAZE.h"

#import <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>

#include <dirent.h>
#include <list>

#include <hdf5.h>

#define GM_COUNT    40
#define SIFT_DIMENSION  128
#define FV_DIMENSION 2*GM_COUNT*128

#define MAX_DIMENSION   44

#define IMG_WIDTH   640
#define IMG_HEIGHT  840

#define X_MIN   32
#define X_MAX   608
#define Y_MIN   42
#define Y_MAX   798

#define SIFT_COUNT  0

#define EMBED_DIR

using namespace cv;
using namespace std;

void readDirectory(const char *directoryName, std::vector<std::string>& filenames, int searchFolder);

void saveMatrix(float *matrix, int rows, int cols, const char *fileName);

void saveMatrix(void *matrix, int dataType, int rows, int cols, const char *fileName);

Mat readMatrix(const char *fileName);

void *readMatrix(const char *fileName, int dataType);

void calcMeanVectorOfMatrix(const char *fileName, float **mean, int &len);

void saveGmmModel(const char *filename, float *means, float *covariances, float *priors);

void getWholeAkazeDescriptorAndVectorFromDir(const char *rootDir, vector<Mat> &wholeVector, Mat &wholeData);

void getWholeDescriptorAndVectorFromDir(const char *rootDir, vector<Mat> &wholeVector, Mat &wholeData);

void siftDescriptorToTrain(const char *rootDir);

//wholeData.rows is the count of sift descriptors, whole.cols = 128
void trainEM(Mat wholeData, const char *savePath, int gmm_count, int iterations);

void readGmmModel(const char *filename, float **means, float **covariances, float **priors);

Mat computeFisherVector(Mat descriptors, float *means, float *covariances, float *priors);

void cvtToRootSift(Mat descriptors);

#endif /* comm_hpp */
