#include <string>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/mpi/collectives.hpp>

#include "mpiSfM/SIFT_Matching.hpp"

using namespace std;
namespace mpi = boost::mpi;

int main( int argc, char * argv[] )
{
	mpi::environment env;
	mpi::communicator world;

	vector<vector<int> > currentCoreFrameOrder;
	vector<FrameCamera> vFC;
	CvMat *K = cvCreateMat(3, 3, CV_32FC1);
	CvMat *invK = cvCreateMat(3, 3, CV_32FC1);
	double focal_x = 0;
	double focal_y = 0;
	double princ_x = 0;
	double princ_y = 0;
	double omega = 0;
	int display = 0;

	string savefile_static;
	if (world.rank() == 0) {

		string path = FILE_PATH;

		cv::FileStorage fs;

		fs.open("data/match_keypoints_info.yml", cv::FileStorage::READ);

		string savepath = (string)fs["match_keypoints_demo_settings"]["reconstruction_folder"];
		string savepath_m = (string)fs["match_keypoints_demo_settings"]["measurement_folder"];
		string keylist = (string)fs["general_settings"]["keys_folder"];
		string imagelist = (string)fs["general_settings"]["image_folder"];
		savefile_static = savepath_m + "static_measurement_desc%07d.txt";

		vector<int> vAddedCameraID;
		FileName fn;

		vector<Camera> vCamera;
		vector<int> vFirstFrame, vStrideFrame;
		vector<string> vDynamicObjectWindowFile, vPath;

		boost::filesystem::create_directories(savepath.c_str());
		boost::filesystem::create_directories(savepath_m.c_str());

		vector<string> vImagename;
		vector<string> vKeysname;
		LoadKeys(keylist, vKeysname);
		LoadImages(imagelist, vImagename);

		for (int ikj = 0; ikj < vImagename.size(); ikj++)
			cout << vImagename[ikj] << ", " << vKeysname[ikj] << endl;

		Camera cam;
		cam.id = 0;
		for (int iFile = 0; iFile < vImagename.size(); iFile++) {
			cam.vImageFileName.push_back(path + imagelist + vImagename[iFile]);
			cam.vKeyFileName.push_back(path + keylist + vKeysname[iFile]);
			cam.vTakenFrame.push_back(iFile);
			cout << *(cam.vImageFileName.end() - 1) << endl;
		}
		vCamera.push_back(cam);

		int im_width = (int)fs["calibration_settings"]["ImageWidth"];
		int im_height = (int)fs["calibration_settings"]["ImageHeight"];
		focal_x = (double)fs["calibration_settings"]["FocalLengthX"];
		focal_y = (double)fs["calibration_settings"]["FocalLengthY"];
		princ_x = (double)fs["calibration_settings"]["PrincipalPointX"];
		princ_y = (double)fs["calibration_settings"]["PrincipalPointY"];
		omega = (double)fs["calibration_settings"]["DistW"];
		double distCtrX = (double)fs["calibration_settings"]["CenterOfDistortionX"];
		double distCtrY = (double)fs["calibration_settings"]["CenterOfDistortionY"];

		fs.release();

		currentCoreFrameOrder.resize(world.size());
		int j_max = currentCoreFrameOrder.size() - 1;
		int j = 0;
		bool alt = false;
		for (int i = 0; i < vCamera[0].vTakenFrame.size(); i++) {
			currentCoreFrameOrder[j].push_back(i);
			if (alt == false) {
				if (j == j_max) alt = !alt;
				else j++;
			}else {
				if (j == 0) alt = !alt;
				else j--;
			}
		}

		if (display)
			cvNamedWindow("SIFT_LOWES", CV_WINDOW_AUTOSIZE);
		vector<int> vTotalTakenFrame;
		vector<int> vCameraID;
		vector<string> vImageFileName;

		vFC.resize(vCamera[0].vTakenFrame.size());

		for (int iFrame = 0; iFrame < vCamera[0].vTakenFrame.size(); iFrame++) {
			FrameCamera fc;
			string keyFile = vCamera[0].vKeyFileName[iFrame];
			fc.imageFileName = vCamera[0].vImageFileName[iFrame];

			fc.cameraID = 0;
			fc.frameIdx = iFrame;

			vector<SIFT_Descriptor> vSift_desc;
			LoadSIFTData_int(keyFile, vSift_desc);

			vector<double> vx1, vy1;
			vector<double> dis_vx1, dis_vy1;

			for (int isift = 0; isift < vSift_desc.size(); isift++) {
				vx1.push_back(vSift_desc[isift].x);
				vy1.push_back(vSift_desc[isift].y);
			}
			Undistortion(omega, distCtrX, distCtrY, vx1, vy1);
			for (int isift = 0; isift < vSift_desc.size(); isift++) {
				vSift_desc[isift].x = vx1[isift];
				vSift_desc[isift].y = vy1[isift];
			}

			fc.vSift_desc = vSift_desc;
			vSift_desc.clear();
			vFC[iFrame] = fc;
		}
	}


	// A series of broadcasts will happen here and we need to take care of them.
	broadcast(world, currentCoreFrameOrder, 0);
	broadcast(world, vFC, 0);
	broadcast(world, savefile_static, 0);
	broadcast(world, omega, 0);
	broadcast(world, focal_x, 0);
	broadcast(world, focal_y, 0);
	broadcast(world, princ_x, 0);
	broadcast(world, princ_y, 0);

	cout << "Rank: " << world.rank() << " ";
	for (auto&ee : currentCoreFrameOrder[world.rank()])
		cout << ee << ", ";
	cout << endl;

	cvSetIdentity(K);
	cvSetReal2D(K, 0, 0, focal_x);
	cvSetReal2D(K, 0, 2, princ_x);
	cvSetReal2D(K, 1, 1, focal_y);
	cvSetReal2D(K, 1, 2, princ_y);
	cvInvert(K, invK);

	vector<int> vFrameOrder = currentCoreFrameOrder[world.rank()];
	for (int iFC = 0; iFC < vFrameOrder.size(); iFC++) {
		int iFC1 = vFrameOrder[iFC];

		FrameCamera cFC = vFC[iFC1];
		vector<Feature> feature_static;
		Iterate_SIFT_STATIC_MP(vFC, iFC1, K, invK, omega, feature_static, display);

		for (int iFeature = 0; iFeature < feature_static.size(); iFeature++)
		     feature_static[iFeature].id = 0;

		char temp[1000];
		sprintf(temp, savefile_static.c_str(), iFC1);
		string savefile_static1 = temp;
		cout<<temp<<endl;
		SaveMeasurementData_RGB_DESC(savefile_static1, feature_static, FILESAVE_WRITE_MODE);
		feature_static.clear();
	}

	// vvSift_desc.clear();
	// vector<int> vnFrame;
	// for (int i = 0; i < vCamera.size(); i++)
	//      vnFrame.push_back(vCamera[i].nFrames);

	return 0;
}
