#ifndef SIFTPP_HPP
#define SIFTPP_HPP


#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdint.h>
#include <memory>
#include <math.h>
#include <mpi.h>

#include "vlfeat_sift/sift.hpp"



#define RAW_CONST_PT(x) reinterpret_cast<char const*>(x)
#define RAW_PT(x)       reinterpret_cast<char*>(x)

// keypoint list
typedef std::vector<std::pair<VL::Sift::Keypoint, VL::float_t> > Keypoints;

void generate_keypoints(std::string image_name, std::string outputkey_filename);
std::ostream& insertDescriptor(std::ostream& os, VL::float_t const * descr_pt, bool binary, bool fp );

#endif //SIFTPP_HPP
