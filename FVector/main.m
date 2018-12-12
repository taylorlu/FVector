//
//  main.m
//  FVector
//
//  Created by LuDong on 2018/1/19.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#import <Foundation/Foundation.h>
#include "generic.h"
#include "mathop.h"
#include "random.h"
#include "gmm.h"
#include "fisher.h"
#include "sift.h"

#import <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>

#include <dirent.h>
#include <list>

#include <hdf5.h>

using namespace cv;

#define GM_COUNT    20
#define SIFT_DIMENSION  128
#define FV_DIMENSION 2*20*128

void readDirectory(const char *directoryName, std::vector<std::string>& filenames, int searchFolder) {
    
    filenames.clear();
    struct dirent *dirp;
    DIR* dir = opendir(directoryName);
    
    while ((dirp = readdir(dir)) != nullptr) {
        if(dirp->d_name[0]!='.') {
            if (dirp->d_type == DT_REG && !searchFolder) {
                // 文件
                std::string directoryStr(directoryName);
                std::string nameStr(dirp->d_name);
                filenames.push_back(directoryStr + "/" + nameStr);
                //printf("file: %s\n", dirp->d_name);
            } else if (dirp->d_type == DT_DIR && searchFolder) {
                // 文件夹
                std::string directoryStr(directoryName);
                std::string nameStr(dirp->d_name);
                filenames.push_back(directoryStr + "/" + nameStr);
                //printf("folder: %s\n", dirp->d_name);
            }
        }
    }
    std::sort(filenames.begin(), filenames.end());
    closedir(dir);
}

void saveMatrix(float *matrix, int rows, int cols, const char *fileName) {
    
    //save fisher vectors(images, so a matrix) in each class folder, (label-00, label-01, ...)
    hid_t fileId = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t dims[] = {(hsize_t)rows, (hsize_t)cols};
    hid_t dataSpaceId = H5Screate_simple(2, dims, NULL);
    hid_t dataSetId = H5Dcreate1(fileId, "fileName", H5T_NATIVE_FLOAT, dataSpaceId, H5P_DEFAULT);
    H5Dwrite(dataSetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix);

    H5Dclose(dataSetId);
    H5Sclose(dataSpaceId);
    H5Fclose(fileId);
}

void saveGmmModel(float *means, float *covariances, float *priors) {
    
    //save gmm model training from all sift descriptors in whole images data
    hid_t fileId = H5Fcreate("/Users/ludong/Desktop/model10/gmm20_sp10.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    hsize_t dims_means[] = {(hsize_t)GM_COUNT, (hsize_t)SIFT_DIMENSION};
    hid_t dataSpaceId = H5Screate_simple(2, dims_means, NULL);
    hid_t dataSetId = H5Dcreate1(fileId, "/means", H5T_NATIVE_FLOAT, dataSpaceId, H5P_DEFAULT);
    herr_t status = H5Dwrite(dataSetId, H5T_NATIVE_FLOAT, H5S_ALL, dataSpaceId, H5P_DEFAULT, means);
    
    hsize_t dims_convs[] = {(hsize_t)GM_COUNT, (hsize_t)SIFT_DIMENSION};
    hid_t dataSpaceId2 = H5Screate_simple(2, dims_convs, NULL);
    hid_t dataSetId2 = H5Dcreate1(fileId, "/covs", H5T_NATIVE_FLOAT, dataSpaceId2, H5P_DEFAULT);
    status = H5Dwrite(dataSetId2, H5T_NATIVE_FLOAT, H5S_ALL, dataSpaceId2, H5P_DEFAULT, covariances);
    
    hsize_t dims_priors[] = {(hsize_t)GM_COUNT};
    hid_t dataSpaceId3 = H5Screate_simple(1, dims_priors, NULL);
    hid_t dataSetId3 = H5Dcreate1(fileId, "/priors", H5T_NATIVE_FLOAT, dataSpaceId3, H5P_DEFAULT);
    status = H5Dwrite(dataSetId3, H5T_NATIVE_FLOAT, H5S_ALL, dataSpaceId3, H5P_DEFAULT, priors);

    H5Dclose(dataSetId);
    H5Sclose(dataSpaceId);
    H5Dclose(dataSetId2);
    H5Sclose(dataSpaceId2);
    H5Dclose(dataSetId3);
    H5Sclose(dataSpaceId3);
    H5Fclose(fileId);
}

void trainEM(Mat wholeData) {

    printf("Begin EM...\r\n");
    float * means ;
    float * covariances ;
    float * priors ;
    int iterations = 200;

    // create a new instance of a GMM object for float data
    VlGMM *gmm = vl_gmm_new (VL_TYPE_FLOAT, SIFT_DIMENSION, GM_COUNT);
    // set the maximum number of EM iterations to 1000
    vl_gmm_set_max_num_iterations (gmm, iterations);
    // set the initialization to random selection
    vl_gmm_set_initialization (gmm, VlGMMRand);
    // cluster the data, i.e. learn the GMM

    int numData = wholeData.rows;
    void *data = wholeData.data;

    vl_gmm_cluster (gmm, data, numData);
    // get the means, covariances, and priors of the GMM
    means = (float *)vl_gmm_get_means(gmm);
    covariances = (float *)vl_gmm_get_covariances(gmm);//only saved the diagonal values
    priors = (float *)vl_gmm_get_priors(gmm);

    saveGmmModel(means, covariances, priors);
    printf("End EM...\r\n");
}

void readModel(float **means, float **covariances, float **priors) {

    hid_t fileId = H5Fopen("/Users/ludong/Desktop/model10/gmm20_sp10.h5", H5F_ACC_RDONLY, H5P_DEFAULT);

    ////get dataset 1
    hid_t datasetId = H5Dopen1(fileId, "/means");
    hid_t spaceId = H5Dget_space(datasetId);
    int ndims = H5Sget_simple_extent_ndims(spaceId);
    hsize_t dims[ndims];
    herr_t status = H5Sget_simple_extent_dims(spaceId, dims, NULL);

    int cap = 1;
    for(int i=0; i<ndims; i++) {
        cap *= dims[i];
    }
    *means = (float *)malloc(sizeof(float)*cap);
    hid_t memspace = H5Screate_simple(ndims,dims,NULL);
    status = H5Dread(datasetId, H5T_NATIVE_FLOAT, memspace, spaceId, H5P_DEFAULT, *means);

    status = H5Sclose(spaceId);
    status = H5Dclose(datasetId);
    
    ////get dataset 2
    datasetId = H5Dopen1(fileId, "/covs");
    spaceId = H5Dget_space(datasetId);
    ndims = H5Sget_simple_extent_ndims(spaceId);
    hsize_t dims2[ndims];
    status = H5Sget_simple_extent_dims(spaceId, dims2, NULL);

    cap = 1;
    for (int i=0; i<ndims; i++) {
        cap *= dims2[i];
    }
    *covariances = (float *)malloc(sizeof(float)*cap);
    memspace = H5Screate_simple(ndims,dims2,NULL);
    status = H5Dread(datasetId, H5T_NATIVE_FLOAT, memspace, spaceId, H5P_DEFAULT, *covariances);

    H5Sclose(spaceId);
    H5Dclose(datasetId);

    ////get dataset 3
    datasetId = H5Dopen1(fileId, "/priors");
    spaceId = H5Dget_space(datasetId);
    ndims = H5Sget_simple_extent_ndims(spaceId);
    hsize_t dims3[ndims];
    status = H5Sget_simple_extent_dims(spaceId, dims3, NULL);
    
    cap = 1;
    for (int i=0; i<ndims; i++) {
        cap *= dims3[i];
    }
    *priors = (float *)malloc(sizeof(float)*cap);
    memspace = H5Screate_simple(ndims,dims3,NULL);
    status = H5Dread(datasetId, H5T_NATIVE_FLOAT, memspace, spaceId, H5P_DEFAULT, *priors);
    
    H5Sclose(spaceId);
    H5Dclose(datasetId);
    H5Fclose(fileId);
}

void descriptorToTrain() {//gather sift vectors from all pictures, and train gmm, save gmm model

    const char *rootDir = "/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples10";
    std::vector<std::string> folders;
    
    readDirectory(rootDir, folders, 1);
    Mat wholeData;
    for(int i=0; i<folders.size(); i++) {
        printf("%s\n", folders[i].c_str());
        Mat classData;
        std::vector<std::string> files;
        readDirectory(folders[i].c_str(), files, 0);
        for(int j=0;j<files.size();j++) {
            //printf("%s\n", files[j].c_str());
            Mat imgMat(cvLoadImage(files[j].c_str()));
            SiftDescriptorExtractor detector;
            vector<KeyPoint> keypoints;
            detector.detect(imgMat, keypoints);
            Mat descriptors;
            detector.compute(imgMat, keypoints, descriptors);
            wholeData.push_back(descriptors);
        }
    }
    printf("EM rows = %d\r\n", wholeData.rows);
    trainEM(wholeData);
}

void descriptorGmmToFisherVector(float *means, float *covariances, float *priors) {
    //according to gmm, compute fv of each picture, and save to matrix(images in a class folder), each class has a matrix
    //these matries is to train softmax classifier
    
    const char *rootDir = "/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples10";
    std::vector<std::string> folders;
    
    //std::map<std::string, Mat> dicts;   //folder name --> fisher vector(Mat)
    
    readDirectory(rootDir, folders, 1);
    
    for(int i=0; i<folders.size(); i++) {
        printf("%s\n", folders[i].c_str());
        Mat classData;
        std::vector<std::string> files;
        readDirectory(folders[i].c_str(), files, 0);
        for(int j=0;j<files.size();j++) {
//            printf("%s\n", files[j].c_str());
            Mat imgMat(cvLoadImage(files[j].c_str()));
            SiftDescriptorExtractor detector;
            vector<KeyPoint> keypoints;
            detector.detect(imgMat, keypoints);
            Mat descriptors;
            detector.compute(imgMat, keypoints, descriptors);
            
            float *enc = (float *)vl_malloc(sizeof(float) * FV_DIMENSION);
            vl_fisher_encode(enc, VL_TYPE_FLOAT, means, SIFT_DIMENSION, GM_COUNT, covariances, priors, descriptors.data, descriptors.rows, VL_FISHER_FLAG_IMPROVED);
            Mat fisherVector;   //one picture's fisher vector
            
            fisherVector.create(1, FV_DIMENSION, CV_32F);
            memcpy(fisherVector.data, enc, sizeof(float) * FV_DIMENSION);
            classData.push_back(fisherVector);
        }
        string::size_type pos=folders[i].rfind('/');
        std::string path = "/Users/ludong/Desktop/model10/";
        path.append(folders[i].substr(pos==string::npos?folders[i].length():pos+1));
        saveMatrix((float *)classData.data, classData.rows, classData.cols, path.c_str());
    }
}

Mat softmaxW(int labelCount, const char *path) {
    
    //get softmax's weights from hdf5, which train in python
    hid_t fileId = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    
    ////get dataset 1
    hid_t datasetId = H5Dopen1(fileId, "softmax");
    hid_t spaceId = H5Dget_space(datasetId);
    int ndims = H5Sget_simple_extent_ndims(spaceId);
    hsize_t dims[ndims];
    herr_t status = H5Sget_simple_extent_dims(spaceId, dims, NULL);
    
    int cap = 1;
    for(int i=0; i<ndims; i++) {
        cap *= dims[i];
    }
    float *weights = (float *)malloc(sizeof(float)*cap);
    hid_t memspace = H5Screate_simple(ndims,dims,NULL);
    status = H5Dread(datasetId, H5T_NATIVE_FLOAT, memspace, spaceId, H5P_DEFAULT, weights);
    
    Mat weightMat(FV_DIMENSION, labelCount, CV_32F, weights);
    
    status = H5Sclose(spaceId);
    status = H5Dclose(datasetId);
    H5Fclose(fileId);
    return weightMat;
}

Mat computeFisherVector(Mat descriptors, float *means, float *covariances, float *priors) {

    //input gmm model and sift descriptors, calculate fisher vector of a picture
    float *enc = (float *)vl_malloc(sizeof(float) * FV_DIMENSION);
    vl_fisher_encode(enc, VL_TYPE_FLOAT, means, SIFT_DIMENSION, GM_COUNT, covariances, priors, descriptors.data, descriptors.rows, VL_FISHER_FLAG_IMPROVED);
    Mat fisherVector;   //one picture's fisher vector
    fisherVector.create(1, FV_DIMENSION, CV_32F);
    memcpy(fisherVector.data, enc, sizeof(float) * FV_DIMENSION);

    return fisherVector;
}

void saveFile(uint8_t *data, int len, const char *filename) {

    string path("/Users/ludong/Desktop/");
    FILE *file = fopen(path.append(filename).c_str(), "wb");
    fwrite(data, len, 1, file);
    fclose(file);
}

void test() {
    
    float *means = NULL, *covariances = NULL, *priors = NULL;
    readModel(&means, &covariances, &priors);
    
    Mat weightMat = softmaxW(19, "/Users/ludong/Desktop/model/softmax.pkl");
    
    const char *folder = "/Users/ludong/Desktop/pageScanner/pageScanner/verifySamples";

    std::vector<std::string> files;
    readDirectory(folder, files, 0);
    
    for(int j=0;j<files.size();j++) {
        printf("%s\n", files[j].c_str());
        Mat imgMat2(cvLoadImage(files[j].c_str()));
        
        cvtColor(imgMat2, imgMat2, CV_BGR2GRAY);
        Mat imgMat = imgMat2.clone();
        
//        uint8_t *my = (uint8_t *)imgMat.data;
//        saveFile(my, 840*640, "test2.yuv");
        
        SiftDescriptorExtractor detector;
        vector<KeyPoint> keypoints;
        detector.detect(imgMat, keypoints);
        Mat descriptors;
        detector.compute(imgMat, keypoints, descriptors);
    
        Mat fisherVector = computeFisherVector(descriptors, means, covariances, priors);
        
        float *haha = (float *)fisherVector.data;
        for(int i=1000; i<1200;i++) {
            printf("haha = %f\n", haha[i]);
        }
    
        Mat result = fisherVector*weightMat;

        float *data = (float *)result.data;
        float privot = data[0];
        int index = 0;
        for(int i=1;i<result.cols;i++) {
            float *curData = (float *)(result.data + i*result.step[1]);
            if(*curData>privot) {
                privot = *curData;
                index = i;
            }
        }
        printf("index = %d\n", index);
    }

}

int main(int argc, const char * argv[]) {

//    descriptorToTrain();
//    return 0;

    test();
    return 0;

    float *means = NULL, *covariances = NULL, *priors = NULL;
    readModel(&means, &covariances, &priors);

    descriptorGmmToFisherVector(means, covariances, priors);
    return 0;
    
    Mat weightMat = softmaxW(19, "/Users/ludong/Desktop/model/softmax.pkl");
    
    float *data = (float *)weightMat.data;
    for(int i=0;i<100; i++) {
        printf("%f\n", data[i]);
    }
    
    const char *folder = "/Users/ludong/Desktop/pageScanner/pageScanner/trainSamples/trainSamples10/label-18";
//    char *name = "/Users/ludong/Desktop/pageScanner/pageScanner/testSamples/tt16.jpg";
    std::vector<std::string> files;
    readDirectory(folder, files, 0);

    for(int j=0;j<files.size();j++) {
//        printf("filename = %s\n", files[j].c_str());
        Mat imgMat(cvLoadImage(files[j].c_str()));
        SiftDescriptorExtractor detector;
        vector<KeyPoint> keypoints;
        detector.detect(imgMat, keypoints);
        Mat descriptors;
        detector.compute(imgMat, keypoints, descriptors);
        Mat fisherVector = computeFisherVector(descriptors, means, covariances, priors);
        
        Mat result = fisherVector*weightMat;

        float *data = (float *)result.data;
        float privot = data[0];
        int index = 0;
        for(int i=1;i<result.cols;i++) {
            float *curData = (float *)(result.data + i*result.step[1]);
            if(*curData>privot) {
                privot = *curData;
                index = i;
            }
        }
        printf("index = %d\n", index);
    }
    return 0;

}
