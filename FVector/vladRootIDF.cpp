//
//  vladRootIDF.cpp
//  FVector
//
//  Created by LuDong on 2018/2/9.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#include "vladRootIDF.hpp"

//root node has 3 centers, the penultimate layer node also has 3 centers, traverse only reach penultimate layer
//the center(uint_8) (e.g.:122) but to calculate mean, so change to uint32 in function vl_ikm_get_centers(e.g.:122,0,0,0)
void traverseTree(VlHIKMNode *node, int &subTrees, int *hikmCenters, int &count) {
    
    if(node) {
        const vl_int32 *centers = vl_ikm_get_centers(node->filter);
        for(int i=0; i<subTrees; i++) {
            memcpy(hikmCenters+count*SIFT_DIMENSION, centers+i*SIFT_DIMENSION, SIFT_DIMENSION*sizeof(vl_int32));
            count++;
        }
        if(node->children) {
            for(int k=0; k<vl_ikm_get_K(node->filter); k++) {
                traverseTree(node->children[k], subTrees, hikmCenters, count);
            }
        }
    }
}

//get the leaf node of kd-tree, and store to leafCenters, by index, vl_int32-->float, due to vlad
void getLeafCenters(VlHIKMNode *node, int &subTrees, float *leafCenters, int &count, vl_size height) {
    
    if(node) {
        const vl_int32 *centers = vl_ikm_get_centers(node->filter);
        
        if(height==1) {
            for(int i=0; i<subTrees; i++) {
                for(int j=0; j<SIFT_DIMENSION; j++) {
                    leafCenters[count*SIFT_DIMENSION + j] = (float)centers[i*SIFT_DIMENSION + j];
                }
                count++;
            }
            return;
        }
        else {
            if(node->children) {
                vl_size K = vl_ikm_get_K(node->filter);
                for(int k=0; k<K; k++) {
                    getLeafCenters(node->children[k], subTrees, leafCenters, count, height-1);
                }
            }
        }
    }
}

//init hikmTree using center data saved in HDF5
VlHIKMNode *
initTreeUsingCenters (VlHIKMTree *tree,
        vl_int32 *data,
        int &count, vl_size subTrees, vl_size height)
{
    VlHIKMNode *node = (VlHIKMNode *)vl_malloc (sizeof(VlHIKMNode)) ;
    node->filter = vl_ikm_new (tree->method) ;
    node->filter->K = subTrees;
    node->filter->M = SIFT_DIMENSION;
    node->filter->centers = (vl_int32 *)vl_malloc(sizeof(vl_int32) * SIFT_DIMENSION * subTrees) ;
    
    for(int i=0; i<subTrees; i++) {
        memcpy(node->filter->centers+i*SIFT_DIMENSION, data+count*SIFT_DIMENSION, SIFT_DIMENSION*sizeof(vl_int32));
        count++;
    }
    node->children = (height == 1) ? 0 : (VlHIKMNode **)vl_malloc (sizeof(*node->children) * subTrees) ;
    
    /* recursively process each child */
    if (height > 1) {
        for (vl_uindex k = 0 ; k < subTrees ; ++k) {
            node->children[k] = initTreeUsingCenters(tree, data, count, subTrees, height - 1) ;
        }
    }
    return node ;
}

#if 1

int main() {
    
//    const char *sampleDir = "/Users/ludong/Desktop/pageSamples/trainSamples/trainSamples5";
//    const char *hikmFile = "/Users/ludong/Desktop/model/hikmCenter.h5";
//    const char *idfFile = "/Users/ludong/Desktop/model/idf.h5";
//    const char *vladFile = "/Users/ludong/Desktop/model/standardVlads.h5";
    
    const char *sampleDir = "/Users/ludong/Desktop/ifset/pics";
    const char *hikmFile = "/Users/ludong/Desktop/model/stand/hikmCenter.h5";
    const char *idfFile = "/Users/ludong/Desktop/model/stand/idf.h5";
    const char *vladFile = "/Users/ludong/Desktop/model/stand/standardVlads.h5";
    
    //3 subTree, 5 depth.
    int subTrees = 3;
    int depth = 5;
    int nclusters = pow(subTrees, depth);
    int totalNodes = (pow(subTrees,depth+1)-subTrees)/(subTrees-1);
    int count = 0;
    VlHIKMTree *hikmTree = vl_hikm_new(VL_IKM_LLOYD);
    vl_hikm_init(hikmTree, SIFT_DIMENSION, subTrees, depth);
    vl_hikm_set_max_niters(hikmTree, 1000);
    
    Mat wholeData;
    vector<Mat> wholeVector;
    getWholeAkazeDescriptorAndVectorFromDir(sampleDir, wholeVector, wholeData);
//    getWholeDescriptorAndVectorFromDir(sampleDir, wholeVector, wholeData);//origin sift is float.000
    Mat ucharWholeData;
    wholeData.convertTo(ucharWholeData, CV_8U); //need to be uint8 type, due to hi(nterger)km
    
    Mat imgDescs;
    uchar imgAsgns[wholeVector.size()][nclusters];
    memset(imgAsgns, 0, wholeVector.size()*nclusters*sizeof(uchar));
    
#define MAX_SIFT_COUNT 10000
    vl_uint32 *indexes = (vl_uint32 *)vl_malloc(sizeof(vl_uint32) * MAX_SIFT_COUNT); //store cluster.index of each sift

#define STEP1
#ifdef STEP1
    ////Step 1: train the hikm tree, also calculate the idf of each cluster(leaf), save hikm tree center and idf array
    
    vl_hikm_train(hikmTree, ucharWholeData.data, ucharWholeData.rows);

    int *hikmCenters = (int *)malloc(totalNodes*SIFT_DIMENSION*sizeof(int));
    traverseTree(hikmTree->root, subTrees, hikmCenters, count);
    saveMatrix(hikmCenters, H5T_NATIVE_INT32, totalNodes, SIFT_DIMENSION, hikmFile);//save hikm centers
    free(hikmCenters);

    for(int imgIdx=0; imgIdx<wholeVector.size(); imgIdx++) {

        wholeVector[imgIdx].convertTo(imgDescs, CV_8U);

        vl_uint32 asgn[depth*imgDescs.rows];
        vl_hikm_push(hikmTree, asgn, imgDescs.data, imgDescs.rows);
        //an image ==> 2000 sift descriptors ==> 2000 * path[0,1,0,2...]

        for(int descIdx=0; descIdx<imgDescs.rows; descIdx++) {  //each path[0,1,2,0,1]

            int index = 0;
            for(int i=0; i<depth; i++) {
                vl_uint32 sgn = asgn[descIdx*depth+i];
                int multy = (int)pow(subTrees, depth-i-1);
                index += multy * sgn;       //path of each sift
            }
            indexes[descIdx] = index;       //cur image sift's index
            imgAsgns[imgIdx][index] = 1;    //each image, Asgns's tf=1 indicate has this word
        }
    }

    float idf[nclusters];   //calculate idf as weight of each cluster
    for(int i=0; i<nclusters; i++) {
        idf[i] = 0;
    }
    for(int col=0; col<nclusters; col++) { //3^5 = 243 clusters
        for(int row=0; row<wholeVector.size(); row++) {
            idf[col] += imgAsgns[row][col];
        }
    }
    for(int i=0; i<nclusters; i++) {    //log(1+|D|/Ni)
        idf[i] = log(1+wholeVector.size()/idf[i]);
    }
    saveMatrix(idf, 1, nclusters, idfFile);   //save idf weights array
    
#else
    ////Step 2: load hikm centers and idf, also extract the leaf node centers
    ////Step 3: calculate the vlad vectors of all images
    float *assignments = (float *)malloc(sizeof(float) * MAX_SIFT_COUNT * nclusters);//oneHot vector(idf)
    float *vladEnc = (float *)vl_malloc(sizeof(float) * SIFT_DIMENSION * nclusters);
    float *idf = (float *)readMatrix(idfFile, H5T_NATIVE_FLOAT);

    int *hikmCenters = (int *)readMatrix(hikmFile, H5T_NATIVE_INT32);
    count = 0;
    hikmTree->root = initTreeUsingCenters(hikmTree, hikmCenters, count, subTrees, depth);

    float *leafCenters = (float *)malloc(nclusters*SIFT_DIMENSION*sizeof(float));
    count = 0;
    getLeafCenters(hikmTree->root, subTrees, leafCenters, count, 5);

    Mat classData;

    for(int imgIdx=0; imgIdx<wholeVector.size(); imgIdx++) {

        wholeVector[imgIdx].convertTo(imgDescs, CV_8U);//uint8 to calc hikm path

        vl_uint32 asgn[depth*imgDescs.rows];
        vl_hikm_push(hikmTree, asgn, imgDescs.data, imgDescs.rows);
        //an image ==> 2000 sift descriptors ==> 2000 * path[0,1,0,2...]

        for(int descIdx=0; descIdx<imgDescs.rows; descIdx++) {  //each path[0,1,2,0,1]

            int index = 0;
            for(int i=0; i<depth; i++) {    //calculate index of leaf node
                vl_uint32 sgn = asgn[descIdx*depth+i];
                int multy = (int)pow(subTrees, depth-i-1);
                index += multy * sgn;
            }
            indexes[descIdx] = index;
        }

        memset(assignments, 0, sizeof(float) * imgDescs.rows * nclusters);
        for(int i = 0; i < imgDescs.rows; i++) {  //oneHot vector(idf)
            assignments[i * nclusters + indexes[i]] = 1.;
        }

        Mat imgDescs2;
        wholeVector[imgIdx].convertTo(imgDescs2, CV_32F);//float to calc vlad encoder
        vl_vlad_encode(vladEnc, VL_TYPE_FLOAT, leafCenters, SIFT_DIMENSION, nclusters, imgDescs2.data, imgDescs2.rows, assignments, VL_VLAD_FLAG_UNNORMALIZED);

        for(int i=0; i<nclusters; i++) {  //each cluster
            for(int j=0; j<SIFT_DIMENSION; j++) {
                vladEnc[i*SIFT_DIMENSION + j] *= idf[i];
            }
        }

        //L2 normalize
        float sum = 0;
        for(int vladOne=0; vladOne<SIFT_DIMENSION*nclusters; vladOne++) {
            sum += pow(vladEnc[vladOne], 2);
        }
        sum = sqrt(sum);
        for(int vladOne=0; vladOne<SIFT_DIMENSION*nclusters; vladOne++) {//L2 normalize
            vladEnc[vladOne] = vladEnc[vladOne]/sum;
        }

        //push vlad vector to Mat
        Mat vladVector;   //one picture's vlad vector
        vladVector.create(1, SIFT_DIMENSION * nclusters, CV_32F);
        memcpy(vladVector.data, vladEnc, sizeof(float) * SIFT_DIMENSION * nclusters);
        classData.push_back(vladVector);
    }

    //save vlad vector of each image
    saveMatrix((float *)classData.data, (int)wholeVector.size(), nclusters*SIFT_DIMENSION, vladFile);

    //free memory
    free(vladEnc);
    free(leafCenters);
    free(indexes);
    free(assignments);
#endif
    return 0;
}
#endif
