# add the binary tree directory to the search path
# for include files
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_BINARY_DIR} )

# add the main library
ADD_LIBRARY( mpiSfM
                SIFT++.cpp
                SIFT_Matching.cpp
                DataUtility.cpp
                MathUtility.cpp
)

TARGET_LINK_LIBRARIES( mpiSfM
                ${OpenCV_LIBS}
                ${MPI_LIBRARIES}
                vlfeat_sift
)
