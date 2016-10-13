#include "mpiSfM/SIFT_Matching.hpp"

using namespace std;

int main( int argc, char * argv[] )
{
	string path = FILE_PATH;

	cv::FileStorage fs;

	fs.open("data/match_keypoints_info.yml", cv::FileStorage::READ);

	string savepath = (string)fs["match_keypoints_demo_settings"]["reconstruction_folder"];
	string savepath_m = (string)fs["match_keypoints_demo_settings"]["measurement_folder"];
	string keylist = (string)fs["general_settings"]["keys_folder"];
	string imagelist = (string)fs["general_settings"]["image_folder"];

	vector<int> vAddedCameraID;
	FileName fn;

	string savefile_static = savepath_m + "static_measurement_desc%07d.txt";

	int display = 0;

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
        double focal_x = (double)fs["calibration_settings"]["FocalLengthX"];
        double focal_y = (double)fs["calibration_settings"]["FocalLengthY"];
        double princ_x = (double)fs["calibration_settings"]["PrincipalPointX"];
        double princ_y = (double)fs["calibration_settings"]["PrincipalPointY"];
        double omega = (double)fs["calibration_settings"]["DistW"];
        double distCtrX = (double)fs["calibration_settings"]["CenterOfDistortionX"];
        double distCtrY = (double)fs["calibration_settings"]["CenterOfDistortionY"];

        fs.release();

        CvMat *K = cvCreateMat(3, 3, CV_32FC1);
        cvSetIdentity(K);
        cvSetReal2D(K, 0, 0, focal_x);
        cvSetReal2D(K, 0, 2, princ_x);
        cvSetReal2D(K, 1, 1, focal_y);
        cvSetReal2D(K, 1, 2, princ_y);
        CvMat *invK = cvCreateMat(3, 3, CV_32FC1);
        cvInvert(K, invK);
        // fin_cal.close();

        vector<int> vFrameOrder;
        if (vCamera[0].vTakenFrame.size() % 2 == 0) {
                for (int i = 0; i < vCamera[0].vTakenFrame.size() / 2; i++) {
                        vFrameOrder.push_back(i);
                        vFrameOrder.push_back(vCamera[0].vTakenFrame.size() - i - 1);
                }
        }else {
                for (int i = 0; i < (vCamera[0].vTakenFrame.size() - 1) / 2; i++) {
                        vFrameOrder.push_back(i);
                        vFrameOrder.push_back(vCamera[0].vTakenFrame.size() - i - 1);
                }
                vFrameOrder.push_back((vCamera[0].vTakenFrame.size() - 1) / 2);
        }

        if (display)
                cvNamedWindow("SIFT_LOWES", CV_WINDOW_AUTOSIZE);
        vector<int> vTotalTakenFrame;
        vector<int> vCameraID;
        vector<string> vImageFileName;
        vector<FrameCamera> vFC;

        vector<vector<SIFT_Descriptor> > vvSift_desc;
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

        int nTotal = vFrameOrder.size();
        int current = 0;
        vector<int> vFrameIdx;
        vFrameIdx.resize(vFrameOrder.size(), 0);

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
                SaveMeasurementData_RGB_DESC(savefile_static1, feature_static, FILESAVE_WRITE_MODE);
                feature_static.clear();

                vFrameIdx[iFC] = 1;
                int count = 0;
                for (int ic = 0; ic < vFrameIdx.size(); ic++)
                        if (vFrameIdx[ic] == 1)
                                count++;
                cout << "Status: " << count << " " << nTotal << endl;
        }

        vvSift_desc.clear();
        vector<int> vnFrame;
        for (int i = 0; i < vCamera.size(); i++)
                vnFrame.push_back(vCamera[i].nFrames);

	return 0;
}
