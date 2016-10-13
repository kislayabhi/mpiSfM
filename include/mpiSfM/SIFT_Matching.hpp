#include <fstream>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <flann/flann.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include "mpiSfM/DataUtility.hpp"
#include "mpiSfM/StructDefinition.hpp"


using namespace std;

#define STATIC true
#define STITCHING false
#define ZERO_DISTANCE 1e+0
#define FILE_PATH ""
#define PI 3.14159265359


void Undistortion(double omega, double DistCtrX, double DistCtrY, vector<double> &vx,  vector<double> &vy);
void Iterate_SIFT_STATIC_MP(vector<FrameCamera> &vFC, int currentFC, CvMat *K, CvMat *invK, double omega, vector<Feature> &feature_static, bool display);
int GetStaticCorrespondences(vector<Point> x1, vector<Point> x2, vector<bool> &vIsInlier);
