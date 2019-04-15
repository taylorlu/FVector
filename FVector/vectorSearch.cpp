//
//  vectorSearch.cpp
//  FVector
//
//  Created by LuDong on 2018/1/31.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#include "vectorSearch.hpp"

#if 0

using namespace cv;

void sortMatches(std::vector<DMatch> matches, int totalRow, int topCount, int *topIndices, double *minDistances) {

    for(int cycle=0; cycle<topCount; cycle++) {
        topIndices[cycle] = 0;
        minDistances[cycle] = 2;
        
        for(int index=0; index<totalRow; index++) {
            
            if(cycle!=0) {
                if(matches[index].distance<minDistances[cycle] && matches[index].distance>minDistances[cycle-1]) {
                    topIndices[cycle] = index;
                    minDistances[cycle] = matches[index].distance;
                }
            }
            else {
                if(matches[index].distance<minDistances[cycle]) {
                    topIndices[cycle] = index;
                    minDistances[cycle] = matches[index].distance;
                }
            }
        }
    }
}

void sortColVector(Mat vector, int topN, int *index) {//sort col vector only topN, return index of topN
    
    float provit = 0;
    float lastLargeValue = 10;
    
    for(int i=0; i<topN; i++) {
        for(int row=0; row<vector.rows; row++) {
            
            float *curData = (float *)(vector.data + row*vector.step[0]);
            if(*curData>provit && *curData<lastLargeValue) {
                index[i] = row;
                provit = *curData;
            }
        }
        lastLargeValue = provit;
        provit = 0;
    }
}

Mat genSiftDescs(const char *filename) {
    
    Mat imgMat(cvLoadImage(filename));
    cvtColor(imgMat, imgMat, CV_BGR2GRAY);
    
    SiftDescriptorExtractor detector(SIFT_COUNT);
    vector<KeyPoint> keypoints;
    detector.detect(imgMat, keypoints);
    Mat descriptors;
    detector.compute(imgMat, keypoints, descriptors);
    return descriptors;
}

void saveSiftDescs(Mat descriptors, const char *fileName) {
    
    //save sift descriptors matrix.
    hid_t fileId = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    hsize_t dims[] = {(hsize_t)descriptors.rows, (hsize_t)descriptors.cols};
    hid_t dataSpaceId = H5Screate_simple(2, dims, NULL);
    hid_t dataSetId = H5Dcreate1(fileId, "fileName", H5T_NATIVE_FLOAT, dataSpaceId, H5P_DEFAULT);
    H5Dwrite(dataSetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, descriptors.data);
    
    H5Dclose(dataSetId);
    H5Sclose(dataSpaceId);
    H5Fclose(fileId);
}

int main() {

    float *means = NULL, *covariances = NULL, *priors = NULL;
    readGmmModel("/Users/ludong/Desktop/model/gmm.h5", &means, &covariances, &priors);
    
    Mat wholeData;
    std::vector<std::string> files;
    readDirectory("/Users/ludong/Desktop/model", files, 0);
    for(int j=0;j<files.size();j++) {
        string::size_type idx = files[j].find("label-");
        if (idx != string::npos) {  //contains string 'label-'
            Mat dataMatrix = readMatrix(files[j].c_str());
            wholeData.push_back(dataMatrix);
        }
    }
    
//    FlannBasedMatcher matcher;
    BFMatcher matcher;
    std::vector<std::string> files2;
    readDirectory("/Users/ludong/Desktop/pageSamples/verifySamples", files2, 0);
    std::vector<DMatch> matches;

    for(int j=0;j<files2.size();j++) {
        if(j%10==0) {
            printf("\n=========\n");
        }
        printf("%s\n", files2[j].c_str());
        Mat imgMat2(cvLoadImage(files2[j].c_str()));

        cvtColor(imgMat2, imgMat2, CV_BGR2GRAY);
        Mat imgMat = imgMat2.clone();

        SiftDescriptorExtractor detector(SIFT_COUNT);
        vector<KeyPoint> keypoints;
        detector.detect(imgMat, keypoints);
        Mat descriptors;
        detector.compute(imgMat, keypoints, descriptors);

        Mat fisherVector = computeFisherVector(descriptors, means, covariances, priors);
        
        transpose(fisherVector, fisherVector);
        Mat ddd = wholeData*fisherVector;
        
        int topN = 5;
        int index[topN];
        sortColVector(ddd, topN, index);
        
//        for(int cnt=0; cnt<topN; cnt++) {
//            printf("%d : %f\n", index[cnt], *(float *)(ddd.data + index[cnt]*ddd.step[0]));
//        }
        if(*(float *)(ddd.data + index[0]*ddd.step[0])>0.4) {
            printf("%d : %f\n", index[0], *(float *)(ddd.data + index[0]*ddd.step[0]));
        }
        else {
            printf("...Not found!\n");
        }
//        break;

//        matcher.match(wholeData, fisherVector, matches);
//
//        int topCount = 5;
//        int topIndices[topCount];
//        double minDistances[topCount];
//        sortMatches(matches, wholeData.rows, topCount, topIndices, minDistances);
//
//        for(int i=0; i<topCount; i++) {
//            printf("dist = %lf,  idx = %d\n", minDistances[i], topIndices[i]);
//        }
    }
    
}
#endif
