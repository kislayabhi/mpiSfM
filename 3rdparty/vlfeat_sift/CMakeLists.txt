cmake_minimum_required(VERSION 2.8)
project(vlfeat_sift)
set(VLSIFT_LIB
                sift.cpp
                sift.hpp
                sift.ipp
                sift-conv.tpp
)
add_library(vlfeat_sift ${VLSIFT_LIB})
