//
//  vladRootIDF.hpp
//  FVector
//
//  Created by LuDong on 2018/2/9.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#ifndef vladRootIDF_hpp
#define vladRootIDF_hpp

#include <stdio.h>

#include "vlad.h"
#include "hikmeans.h"
#include "ikmeans.h"
#include "comm.hpp"

//init hikmTree using center data saved in HDF5
VlHIKMNode *initTreeUsingCenters (VlHIKMTree *tree, vl_int32 *data, int &count, vl_size subTrees, vl_size height);

//get the leaf node of kd-tree, and store to leafCenters, by index, vl_int32-->float, due to vlad
void getLeafCenters(VlHIKMNode *node, int &subTrees, float *leafCenters, int &count, vl_size height);

//root node has 3 centers, the penultimate layer node also has 3 centers, traverse only reach penultimate layer
//the center(uint_8) (e.g.:122) but to calculate mean, so change to uint32 in function vl_ikm_get_centers(e.g.:122,0,0,0)
void traverseTree(VlHIKMNode *node, int &subTrees, int *hikmCenters, int &count);

#endif /* vladRootIDF_hpp */
