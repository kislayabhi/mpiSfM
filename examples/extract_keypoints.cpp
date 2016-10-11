#include "mpiSfM/SIFT++.hpp"
#include <string>

using namespace std;

int main(int argc, char** argv)
{
        string arg1(argv[1]);
        string arg2(argv[2]);
	generate_keypoints(arg1, arg2);
}
