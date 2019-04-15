//
//  main.m
//  FVector
//
//  Created by LuDong on 2018/1/19.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#import <Foundation/Foundation.h>
#include "comm.hpp"

#if 0
using namespace cv;

void siftGmmToFisherVector(const char *rootDir, const char *savePath, float *means, float *covariances, float *priors) {
    //according to gmm, compute fv of each picture, and save to matrix(images in a class folder), each class has a matrix
    //these matries is to train softmax classifier
    
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
            SiftDescriptorExtractor detector(SIFT_COUNT);
            vector<KeyPoint> keypoints;
            detector.detect(imgMat, keypoints);
            Mat descriptors;
            detector.compute(imgMat, keypoints, descriptors);
            
            float *enc = (float *)vl_malloc(sizeof(float) * FV_DIMENSION);
            vl_fisher_encode(enc, VL_TYPE_FLOAT, means, SIFT_DIMENSION, GM_COUNT, covariances, priors, descriptors.data, descriptors.rows, VL_FISHER_FLAG_IMPROVED);
            Mat fisherVector;   //one picture's fisher vector
            
            fisherVector.create(1, FV_DIMENSION, CV_32F);
            memcpy(fisherVector.data, enc, sizeof(float) * FV_DIMENSION);
            free(enc);
            classData.push_back(fisherVector);
        }
        string::size_type pos=folders[i].rfind('/');
        std::string path(savePath);
        path.append("/");
        path.append(folders[i].substr(pos==string::npos?folders[i].length():pos+1));
        saveMatrix((float *)classData.data, classData.rows, classData.cols, path.c_str());
    }
}

void dsiftGmmToFisherVector(const char *imgRootDir, const char *savePath, float *means, float *covariances, float *priors) {
    //according to gmm, compute fv of each picture, and save to matrix(images in a class folder), each class has a matrix
    //these matries is to train softmax classifier
    
    std::vector<std::string> folders;
    float *buf = (float *)malloc(sizeof(float)*IMG_HEIGHT*IMG_WIDTH);
    //std::map<std::string, Mat> dicts;   //folder name --> fisher vector(Mat)
    
    readDirectory(imgRootDir, folders, 1);
    
    for(int i=0; i<folders.size(); i++) {
        printf("%s\n", folders[i].c_str());
        Mat classData;
        std::vector<std::string> files;
        readDirectory(folders[i].c_str(), files, 0);
        for(int j=0;j<files.size();j++) {
            
            Mat imgMat(cvLoadImage(files[j].c_str()));
            cvtColor(imgMat, imgMat, CV_BGR2GRAY);  //perhaps copy to first channel
            
            for(int row=0; row<imgMat.rows; row++) {
                for(int col=0; col<imgMat.cols; col++) {
                    buf[row*imgMat.cols+col] = (float)imgMat.data[row*imgMat.cols+col]/255.0;
                }
            }
            
            VlDsiftFilter *dsiftFilter = vl_dsift_new_basic(IMG_WIDTH, IMG_HEIGHT, 15, 15);
            vl_dsift_set_bounds(dsiftFilter, X_MIN, Y_MIN, X_MAX, Y_MAX);
            
            vl_dsift_process(dsiftFilter, buf);
            int num = vl_dsift_get_keypoint_num(dsiftFilter);
            const float *descriptors = vl_dsift_get_descriptors(dsiftFilter);
            printf("num = %d\n", num);
            
            Mat descriptorMat;
            descriptorMat.create(num, SIFT_DIMENSION, CV_32F);
            memcpy(descriptorMat.data, descriptors, num*SIFT_DIMENSION*sizeof(float));
            
            float *enc = (float *)vl_malloc(sizeof(float) * FV_DIMENSION);
            vl_fisher_encode(enc, VL_TYPE_FLOAT, means, SIFT_DIMENSION, GM_COUNT, covariances, priors, descriptorMat.data, descriptorMat.rows, VL_FISHER_FLAG_IMPROVED);
            Mat fisherVector;   //one picture's fisher vector
            
            fisherVector.create(1, FV_DIMENSION, CV_32F);
            memcpy(fisherVector.data, enc, sizeof(float) * FV_DIMENSION);
            free(enc);
            classData.push_back(fisherVector);
        }
        string::size_type pos=folders[i].rfind('/');
        std::string path(savePath);
        path.append("/");
        path.append(folders[i].substr(pos==string::npos?folders[i].length():pos+1));
        saveMatrix((float *)classData.data, classData.rows, classData.cols, path.c_str());
    }
    free(buf);
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



void saveFile(uint8_t *data, int len, const char *filename) {

    string path("/Users/ludong/Desktop/");
    FILE *file = fopen(path.append(filename).c_str(), "wb");
    fwrite(data, len, 1, file);
    fclose(file);
}

void testSift() {
    
    float *means = NULL, *covariances = NULL, *priors = NULL;
    readGmmModel("/Users/ludong/Desktop/model10/gmm20_sp10.h5", &means, &covariances, &priors);
    
    Mat weightMat = softmaxW(19, "/Users/ludong/Desktop/model/softmax.pkl");
    
    const char *folder = "/Users/ludong/Desktop/pageSamples/verifySamples";
    
    std::vector<std::string> files;
    readDirectory(folder, files, 0);
    
    for(int j=0;j<files.size();j++) {
        printf("%s\n", files[j].c_str());
        Mat imgMat2(cvLoadImage(files[j].c_str()));
        
        cvtColor(imgMat2, imgMat2, CV_BGR2GRAY);
        Mat imgMat = imgMat2.clone();
        
//        uint8_t *my = (uint8_t *)imgMat.data;
//        saveFile(my, 840*640, "test2.yuv");
        
        SiftDescriptorExtractor detector(SIFT_COUNT);
        vector<KeyPoint> keypoints;
        detector.detect(imgMat, keypoints);
        Mat descriptors;
        detector.compute(imgMat, keypoints, descriptors);
        int a = descriptors.type();
        
        Mat fisherVector = computeFisherVector(descriptors, means, covariances, priors);
        
        float *haha = (float *)fisherVector.data;
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

void testDSift() {
    
    float *means = NULL, *covariances = NULL, *priors = NULL;
    readGmmModel("/Users/ludong/Desktop/model5_dsift/gmm20_sp5.h5", &means, &covariances, &priors);
    
    Mat weightMat = softmaxW(19, "/Users/ludong/Desktop/model5_dsift/softmax.pkl");
    
    float *buf = (float *)malloc(sizeof(float)*IMG_HEIGHT*IMG_WIDTH);
    
    const char *folder = "/Users/ludong/Desktop/pageSamples/verifySamples";
    std::vector<std::string> files;
    readDirectory(folder, files, 0);
    
    for(int j=0;j<files.size();j++) {

        if(j%10==0) {
            printf("\n==============\n");
        }
        Mat imgMat(cvLoadImage(files[j].c_str()));
        cvtColor(imgMat, imgMat, CV_BGR2GRAY);  //perhaps copy to first channel
        
        for(int row=0; row<imgMat.rows; row++) {
            for(int col=0; col<imgMat.cols; col++) {
                buf[row*imgMat.cols+col] = (float)imgMat.data[row*imgMat.cols+col]/255.0;
            }
        }
        
        VlDsiftFilter *dsiftFilter = vl_dsift_new_basic(IMG_WIDTH, IMG_HEIGHT, 15, 15);
        vl_dsift_set_bounds(dsiftFilter, X_MIN, Y_MIN, X_MAX, Y_MAX);
        
        vl_dsift_process(dsiftFilter, buf);
        int num = vl_dsift_get_keypoint_num(dsiftFilter);
        const float *descriptors = vl_dsift_get_descriptors(dsiftFilter);
//        printf("num = %d\n", num);
        
        Mat descriptorMat;
        descriptorMat.create(num, SIFT_DIMENSION, CV_32F);
        memcpy(descriptorMat.data, descriptors, num*SIFT_DIMENSION*sizeof(float));
        
        Mat fisherVector = computeFisherVector(descriptorMat, means, covariances, priors);
        
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
    free(buf);
}



void dsiftDescriptorToTrain(const char *rootDir) {
    
    std::vector<std::string> folders;
    
    readDirectory(rootDir, folders, 1);
    Mat wholeData;
    float *buf = (float *)malloc(sizeof(float)*IMG_HEIGHT*IMG_WIDTH);
    
    for(int i=0; i<folders.size(); i++) {
        printf("%s\n", folders[i].c_str());
        Mat classData;
        std::vector<std::string> files;
        readDirectory(folders[i].c_str(), files, 0);
        for(int j=0;j<files.size();j++) {
            //printf("%s\n", files[j].c_str());
            Mat imgMat(cvLoadImage(files[j].c_str()));
            cvtColor(imgMat, imgMat, CV_BGR2GRAY);  //perhaps copy to first channel
//            transpose(imgMat, imgMat);
//            flip(imgMat, imgMat, 1);

            for(int row=0; row<imgMat.rows; row++) {
                for(int col=0; col<imgMat.cols; col++) {
                    buf[row*imgMat.cols+col] = (float)imgMat.data[row*imgMat.cols+col]/255.0;
                }
            }
            
            VlDsiftFilter *dsiftFilter = vl_dsift_new_basic(IMG_WIDTH, IMG_HEIGHT, 15, 15);
            vl_dsift_set_bounds(dsiftFilter, X_MIN, Y_MIN, X_MAX, Y_MAX);
            
            vl_dsift_process(dsiftFilter, buf);
            int num = vl_dsift_get_keypoint_num(dsiftFilter);
            const float *descriptors = vl_dsift_get_descriptors(dsiftFilter);
            printf("num = %d\n", num);
            
            Mat descriptorMat;
            descriptorMat.create(num, SIFT_DIMENSION, CV_32F);
            memcpy(descriptorMat.data, descriptors, num*SIFT_DIMENSION*sizeof(float));
            
            wholeData.push_back(descriptorMat);
        }
    }
    free(buf);
    printf("EM rows = %d\r\n", wholeData.rows);
    trainEM(wholeData, "/Users/ludong/Desktop/model10_dsift/gmm20_sp10.h5", GM_COUNT, 500);
}

int main(int argc, const char * argv[]) {
    
//    float *mean = NULL;
//    int len = 0;
//    const char *folder2 = "/Users/ludong/Desktop/model10";
//    std::vector<std::string> files2;
//    readDirectory(folder2, files2, 0);
//
//    int fileCount = 0;
//    for(int j=0; j<files2.size(); j++) {
//        string::size_type idx = files2[j].find("label-");
//        if (idx != string::npos) {  //contains string 'label-'
//            fileCount++;
//        }
//    }
//
//    float *matrix = (float *)malloc(sizeof(float)*fileCount*5120);
//    int fileIndex = 0;
//    for(int k=0; k<files2.size(); k++) {
//        string::size_type idx = files2[k].find("label-");
//        if (idx != string::npos) {  //contains string 'label-'
//            printf("[]%s\r\n", files2[k].c_str());
//            calcMeanVectorOfMatrix(files2[k].c_str(), &mean, len);
//            memcpy(matrix+fileIndex*len, mean, len);
//            free(mean);
//            fileIndex++;
//        }
//        else {
//            printf("===%s\r\n", files2[k].c_str());
//        }
//    }
//
//    //save fisher vectors(images, so a matrix) in each class folder, (label-00, label-01, ...)
//    hid_t fileId = H5Fcreate("/Users/ludong/Desktop/model10/classVector.hdf5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
//
//    hsize_t dims[] = {(hsize_t)fileCount, (hsize_t)len};
//    hid_t dataSpaceId = H5Screate_simple(2, dims, NULL);
//    hid_t dataSetId = H5Dcreate1(fileId, "fileName", H5T_NATIVE_FLOAT, dataSpaceId, H5P_DEFAULT);
//    H5Dwrite(dataSetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix);
//    free(matrix);
//
//    H5Dclose(dataSetId);
//    H5Sclose(dataSpaceId);
//    H5Fclose(fileId);
//
//    return 0;
    
//    siftDescriptorToTrain("/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples5");
//    dsiftDescriptorToTrain("/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples5");
//    return 0;
//

    testDSift();
//    testSift();
    return 0;

//    dsiftGmmToFisherVector("/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples5", "/Users/ludong/Desktop/model5_dsift", means, covariances, priors);
////    siftGmmToFisherVector("/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples5", "/Users/ludong/Desktop/model10", means, covariances, priors);
//    return 0;
    

    return 0;

}
#endif
