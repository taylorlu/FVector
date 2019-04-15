//
//  comm.cpp
//  FVector
//
//  Created by LuDong on 2018/1/26.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#include "comm.hpp"

using namespace cv;

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

void saveMatrix(void *matrix, int dataType, int rows, int cols, const char *fileName) {
    
    //save fisher vectors(images, so a matrix) in each class folder, (label-00, label-01, ...)
    hid_t fileId = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    hsize_t dims[] = {(hsize_t)rows, (hsize_t)cols};
    hid_t dataSpaceId = H5Screate_simple(2, dims, NULL);
    hid_t dataSetId = H5Dcreate1(fileId, "fileName", H5T_NATIVE_FLOAT, dataSpaceId, H5P_DEFAULT);
    H5Dwrite(dataSetId, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix);
    
    H5Dclose(dataSetId);
    H5Sclose(dataSpaceId);
    H5Fclose(fileId);
}

Mat readMatrix(const char *fileName) {
    
    //read fisher vectors(label-00, label-01, ...) to matrix.
    hid_t fileId = H5Fopen(fileName, H5F_ACC_RDONLY, H5P_DEFAULT);
    
    ////get dataset 1
    hid_t datasetId = H5Dopen1(fileId, "fileName");
    hid_t spaceId = H5Dget_space(datasetId);
    int ndims = H5Sget_simple_extent_ndims(spaceId);
    hsize_t dims[ndims];
    herr_t status = H5Sget_simple_extent_dims(spaceId, dims, NULL);
    
    int cap = 1;
    int rows = (int)dims[0];
    int cols = (int)dims[1];
    cap = rows*cols;
    
    Mat dataMatrix;
    dataMatrix.create(rows, cols, CV_32F);
    
    hid_t memspace = H5Screate_simple(ndims, dims, NULL);
    status = H5Dread(datasetId, H5T_NATIVE_FLOAT, memspace, spaceId, H5P_DEFAULT, dataMatrix.data);
    
    H5Sclose(spaceId);
    H5Dclose(datasetId);
    H5Fclose(fileId);
    
    return dataMatrix;
}

void *readMatrix(const char *fileName, int dataType) {
    
    //read fisher vectors(label-00, label-01, ...) to matrix.
    hid_t fileId = H5Fopen(fileName, H5F_ACC_RDONLY, H5P_DEFAULT);
    
    ////get dataset 1
    hid_t datasetId = H5Dopen1(fileId, "fileName");
    hid_t spaceId = H5Dget_space(datasetId);
    int ndims = H5Sget_simple_extent_ndims(spaceId);
    hsize_t dims[ndims];
    herr_t status = H5Sget_simple_extent_dims(spaceId, dims, NULL);
    
    int cap = 1;
    int rows = (int)dims[0];
    int cols = (int)dims[1];
    cap = rows*cols;
    
    void *data = malloc(rows*cols*4);
    
    hid_t memspace = H5Screate_simple(ndims, dims, NULL);
    status = H5Dread(datasetId, dataType, memspace, spaceId, H5P_DEFAULT, data);
    
    H5Sclose(spaceId);
    H5Dclose(datasetId);
    H5Fclose(fileId);
    
    return data;
}

void calcMeanVectorOfMatrix(const char *fileName, float **mean, int &len) {
    
    //read a class of image's matrix(fisher vectors), calculate the mean vector of these vectors
    hid_t fileId = H5Fopen(fileName, H5F_ACC_RDONLY, H5P_DEFAULT);
    
    ////get dataset 1
    hid_t datasetId = H5Dopen1(fileId, "fileName");
    hid_t spaceId = H5Dget_space(datasetId);
    int ndims = H5Sget_simple_extent_ndims(spaceId);
    hsize_t dims[ndims];
    herr_t status = H5Sget_simple_extent_dims(spaceId, dims, NULL);
    
    int cap = 1;
    int rows = (int)dims[0];
    int cols = (int)dims[1];
    cap = rows*cols;
    float *matrix = (float *)malloc(sizeof(float)*cap);
    hid_t memspace = H5Screate_simple(ndims, dims, NULL);
    status = H5Dread(datasetId, H5T_NATIVE_FLOAT, memspace, spaceId, H5P_DEFAULT, matrix);
    
    len = cols;
    *mean = (float *)malloc(sizeof(float)*cols);
    float *meanTmp = *mean;
    for (int i=0; i<cols; i++) {
        float total = 0;
        for (int k=0; k<rows; k++) {
            total += *(matrix + k*cols + i);
        }
        meanTmp[i] = total/rows;
    }
    
    H5Sclose(spaceId);
    H5Dclose(datasetId);
    H5Fclose(fileId);
    free(matrix);
}

void saveGmmModel(const char *filename, float *means, float *covariances, float *priors) {
    
    //save gmm model training from all sift descriptors in whole images data
    hid_t fileId = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
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

void trainEM(Mat wholeData, const char *savePath, int gmm_count, int iterations) {
    
    printf("Begin EM...\r\n");
    float * means ;
    float * covariances ;
    float * priors ;
    
    // create a new instance of a GMM object for float data
    VlGMM *gmm = vl_gmm_new (VL_TYPE_FLOAT, SIFT_DIMENSION, gmm_count);
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
    for(int i=0; i<gmm_count; i++)  {
        printf("%f\n", priors[i]);
    }
    saveGmmModel(savePath, means, covariances, priors);
    printf("End EM...\r\n");
}

void readGmmModel(const char *filename, float **means, float **covariances, float **priors) {
    
    hid_t fileId = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    
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

Mat computeFisherVector(Mat descriptors, float *means, float *covariances, float *priors) {
    
    //input gmm model and sift descriptors, calculate fisher vector of a picture
    float *enc = (float *)vl_malloc(sizeof(float) * FV_DIMENSION);
    vl_fisher_encode(enc, VL_TYPE_FLOAT, means, SIFT_DIMENSION, GM_COUNT, covariances, priors, descriptors.data, descriptors.rows, VL_FISHER_FLAG_IMPROVED);
    Mat fisherVector;   //one picture's fisher vector
    fisherVector.create(1, FV_DIMENSION, CV_32F);
    memcpy(fisherVector.data, enc, sizeof(float) * FV_DIMENSION);
    
    return fisherVector;
}

Mat getWholeSiftDescriptorsFromDir(const char *rootDir) {

    Mat wholeData;

    std::vector<std::string> files;
    readDirectory(rootDir, files, 0);
    for(int j=0;j<files.size();j++) {
        IplImage *image = cvLoadImage(files[j].c_str());
        if(image==NULL)
            continue;
        Mat imgMat(image);
        cvtColor(imgMat, imgMat, CV_BGR2GRAY);

        SiftDescriptorExtractor detector(SIFT_COUNT);
        vector<KeyPoint> keypoints;
        detector.detect(imgMat, keypoints);
        Mat descriptors;
        detector.compute(imgMat, keypoints, descriptors);
        wholeData.push_back(descriptors);
    }

    return wholeData;
}

void getWholeAkazeDescriptorAndVectorFromDir(const char *rootDir, vector<Mat> &wholeVector, Mat &wholeData) {
#undef EMBED_DIR
#ifdef EMBED_DIR
    std::vector<std::string> folders;
    readDirectory(rootDir, folders, 1);
    for(int i=0; i<folders.size(); i++) {
        printf("%s\n", folders[i].c_str());
        std::vector<std::string> files;
        readDirectory(folders[i].c_str(), files, 0);
#else
        std::vector<std::string> files;
        readDirectory(rootDir, files, 0);
#endif
        
        for(int j=0;j<files.size();j++) {
            IplImage *image = cvLoadImage(files[j].c_str());
            if(image==NULL)
                continue;
            Mat imgMat = cvarrToMat(image);
            cvtColor(imgMat, imgMat, CV_BGR2GRAY);
            
            vector<KeyPoint> kpts1, kpts2;
            Mat desc1, desc2;
            
            //            Ptr<AKAZE> akaze = AKAZE::create();
            //            akaze->detectAndCompute(imgMat, noArray(), kpts1, desc1);
            //            printf("%d, %d, %d\n", (int)kpts1.size(), desc1.cols, desc1.rows);
            
            AKAZEOptions options = AKAZEOptions();
            libAKAZE::AKAZE evolution(options);
            vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            
            evolution.Create_Nonlinear_Scale_Space(imgMat);
            evolution.Feature_Detection(keypoints);
            evolution.Compute_Descriptors(keypoints, descriptors);
            
            
//            SiftDescriptorExtractor detector(SIFT_COUNT);
//            vector<KeyPoint> keypoints;
//            detector.detect(imgMat, keypoints);
//            Mat descriptors;
//            detector.compute(imgMat, keypoints, descriptors);
            
            wholeData.push_back(descriptors);
            wholeVector.push_back(descriptors);
        }
#ifdef EMBED_DIR
    }
#endif
}

void getWholeDescriptorAndVectorFromDir(const char *rootDir, vector<Mat> &wholeVector, Mat &wholeData) {
    
#ifdef EMBED_DIR
    std::vector<std::string> folders;
    readDirectory(rootDir, folders, 1);
    for(int i=0; i<folders.size(); i++) {
        printf("%s\n", folders[i].c_str());
        std::vector<std::string> files;
        readDirectory(folders[i].c_str(), files, 0);
#else
        std::vector<std::string> files;
        readDirectory(rootDir, files, 0);
#endif
        
        for(int j=0;j<files.size();j++) {
            IplImage *image = cvLoadImage(files[j].c_str());
            if(image==NULL)
                continue;
            Mat imgMat = cvarrToMat(image);
            cvtColor(imgMat, imgMat, CV_BGR2GRAY);
            
            vector<KeyPoint> kpts1, kpts2;
            Mat desc1, desc2;
            
//            Ptr<AKAZE> akaze = AKAZE::create();
//            akaze->detectAndCompute(imgMat, noArray(), kpts1, desc1);
//            printf("%d, %d, %d\n", (int)kpts1.size(), desc1.cols, desc1.rows);
            
            SiftDescriptorExtractor detector(SIFT_COUNT);
            vector<KeyPoint> keypoints;
            detector.detect(imgMat, keypoints);
            Mat descriptors;
            detector.compute(imgMat, keypoints, descriptors);
            
            wholeData.push_back(descriptors);
            wholeVector.push_back(descriptors);
        }
#ifdef EMBED_DIR
    }
#endif
}

void siftDescriptorToTrain(const char *rootDir, const char *savePath) {
    //gather sift vectors from all pictures, and train gmm, save gmm model

    Mat wholeData;

#ifdef EMBED_DIR

    std::vector<std::string> folders;
    readDirectory(rootDir, folders, 1);
    for(int i=0; i<folders.size(); i++) {
        printf("%s\n", folders[i].c_str());
        std::vector<std::string> files;
        readDirectory(folders[i].c_str(), files, 0);
#else
        std::vector<std::string> files;
        readDirectory(rootDir, files, 0);
#endif
        for(int j=0;j<files.size();j++) {
            IplImage *image = cvLoadImage(files[j].c_str());
            if(image==NULL)
                continue;
            Mat imgMat(image);
            cvtColor(imgMat, imgMat, CV_BGR2GRAY);

            SiftDescriptorExtractor detector(SIFT_COUNT);
            vector<KeyPoint> keypoints;
            detector.detect(imgMat, keypoints);
            Mat descriptors;
            detector.compute(imgMat, keypoints, descriptors);
            wholeData.push_back(descriptors);

//            printf("%d, %d\n", descriptors.rows, keypoints.size());
//            String standardDir = "/Users/ludong/Desktop/rrr/";
//            char name[100] = {0};
//            sprintf(name, "standard%02d", j);
//            saveMatrix((float *)descriptors.data, descriptors.rows, descriptors.cols, standardDir.append(name).c_str());
//
//            int ptCount = (int)keypoints.size();
//            float *pointXY = (float *)malloc(ptCount*sizeof(float)*2);
//            for(int i=0; i<keypoints.size(); i++) {
//                pointXY[i] = keypoints[i].pt.x;
//                pointXY[i+ptCount] = keypoints[i].pt.y;
//            }
//            standardDir = "/Users/ludong/Desktop/lkpt/";
//            saveMatrix(pointXY, 2, ptCount, standardDir.append(name).c_str());
        }
#ifdef EMBED_DIR
    }
#endif

    printf("EM rows = %d\r\n", wholeData.rows);
    trainEM(wholeData, savePath, GM_COUNT, 1000);
}

void siftDescriptorToTrain_fromH5Mat(const char *rootDir, const char *savePath) {
    
    Mat wholeData;
    std::vector<std::string> files;
    readDirectory(rootDir, files, 0);
    for(int j=0;j<files.size();j++) {
        Mat mat = readMatrix(files[j].c_str());
        wholeData.push_back(mat);
    }
    trainEM(wholeData, savePath, GM_COUNT, 1000);
}

void savePCA(const string &file_name,cv::PCA &pca_) {
    FileStorage fs(file_name,FileStorage::WRITE);
    fs << "mean" << pca_.mean;
    fs << "e_vectors" << pca_.eigenvectors;
    fs << "e_values" << pca_.eigenvalues;
    fs.release();
}

void loadPCA(const string &file_name,cv::PCA &pca_) {
    FileStorage fs(file_name,FileStorage::READ);
    fs["mean"] >> pca_.mean ;
    fs["e_vectors"] >> pca_.eigenvectors ;
    fs["e_values"] >> pca_.eigenvalues ;
    fs.release();
}

#if 0

int main() {
    
//    const char *gmmModelPath = "/Users/ludong/Desktop/model/gmm.h5";
//    const char *trainImgPath = "/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples5";
//    const char *fvSavePath = "/Users/ludong/Desktop/model/standardFVs.h5";
    
    const char *gmmModelPath = "/Users/ludong/Desktop/model/stand/gmm.h5";
//    const char *trainImgPath = "/Users/ludong/Desktop/frontModel";
    const char *trainImgPath = "/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples5";
    const char *fvSavePath = "/Users/ludong/Desktop/model/stand/standardFVs.h5";
    
#undef STEP1
#ifdef STEP1
//    //step1: gather images' sift descriptors, train GMM model.
    siftDescriptorToTrain(trainImgPath, gmmModelPath);
//    siftDescriptorToTrain_fromH5Mat("/Users/ludong/Desktop/luyu", gmmModelPath);
    
#else
    
    //step2: generate fisher vectors in matrix of images base gmm model.
    float *means = NULL, *covariances = NULL, *priors = NULL;
    readGmmModel(gmmModelPath, &means, &covariances, &priors);
    Mat classData;

//#ifdef EMBED_DIR
//    std::vector<std::string> folders;
//    readDirectory(trainImgPath, folders, 1);
//    for(int i=0; i<folders.size(); i++) {
//        printf("%s\n", folders[i].c_str());
//        std::vector<std::string> files;
//        readDirectory(folders[i].c_str(), files, 0);
//#else
//        std::vector<std::string> files;
//        readDirectory(trainImgPath, files, 0);
//#endif
//        for(int j=0;j<files.size();j++) {
//            IplImage *image = cvLoadImage(files[j].c_str());
//            if(image==NULL)
//                continue;
//            Mat imgMat(image);
//            cvtColor(imgMat, imgMat, CV_BGR2GRAY);
//
//            SiftDescriptorExtractor detector(SIFT_COUNT);
//            vector<KeyPoint> keypoints;
//            detector.detect(imgMat, keypoints);
//            Mat descriptors;
//            detector.compute(imgMat, keypoints, descriptors);
//
//            float *enc = (float *)vl_malloc(sizeof(float) * FV_DIMENSION);
//            vl_fisher_encode(enc, VL_TYPE_FLOAT, means, SIFT_DIMENSION, GM_COUNT, covariances, priors, descriptors.data, descriptors.rows, VL_FISHER_FLAG_IMPROVED);
//            Mat fisherVector;   //one picture's fisher vector
//
//            fisherVector.create(1, FV_DIMENSION, CV_32F);
//            memcpy(fisherVector.data, enc, sizeof(float) * FV_DIMENSION);
//            free(enc);
//            classData.push_back(fisherVector);
//        }
//#ifdef EMBED_DIR
//    }
//#endif
    
    std::vector<std::string> files;
    readDirectory("/Users/ludong/Desktop/rrr", files, 0);
    for(int j=0;j<files.size();j++) {

        Mat descriptors = readMatrix(files[j].c_str());

        float *enc = (float *)vl_malloc(sizeof(float) * FV_DIMENSION);
        vl_fisher_encode(enc, VL_TYPE_FLOAT, means, SIFT_DIMENSION, GM_COUNT, covariances, priors, descriptors.data, descriptors.rows, VL_FISHER_FLAG_IMPROVED);
        Mat fisherVector;   //one picture's fisher vector

        fisherVector.create(1, FV_DIMENSION, CV_32F);
        memcpy(fisherVector.data, enc, sizeof(float) * FV_DIMENSION);
        free(enc);
        classData.push_back(fisherVector);
    }
    
#undef ENABLE_PCA
#ifdef ENABLE_PCA
    PCA pca(classData, Mat(), CV_PCA_DATA_AS_ROW, MAX_DIMENSION);
    Mat compressMat = cvCreateMat(classData.rows, MAX_DIMENSION, classData.type());
    pca.project(classData, compressMat);
    savePCA("/Users/ludong/Desktop/frontModel/pca.h5", pca);
    saveMatrix((float *)compressMat.data, compressMat.rows, compressMat.cols, fvSavePath);
#else
    saveMatrix((float *)classData.data, classData.rows, classData.cols, fvSavePath);
#endif
#endif
    return 0;
}

void cvtToRootSift(Mat descriptors) {
    
}

#endif

