#include "mpiSfM/SIFT_Matching.hpp"


using namespace std;

void Undistortion(double omega, double DistCtrX, double DistCtrY, vector<double> &vx,  vector<double> &vy)
{
	for (int iPoint = 0; iPoint < vx.size(); iPoint++) {
		double x = vx[iPoint] - DistCtrX;
		double y = vy[iPoint] - DistCtrY;
		double r_d = sqrt(x * x + y * y);
		double r_u = tan(r_d * omega) / 2 / tan(omega / 2);
		double x_u = r_u / r_d * x;
		double y_u = r_u / r_d * y;
		vx[iPoint] = x_u + DistCtrX;
		vy[iPoint] = y_u + DistCtrY;
	}
}

void Iterate_SIFT_STATIC_MP(vector<FrameCamera> &vFC, int currentFC, CvMat *K, CvMat *invK, double omega, vector<Feature> &feature_static, bool display)
{
	FrameCamera cFC = vFC[currentFC];

	// IplImage *iplImg1 = cvLoadImage(cFC.imageFileName.c_str());

	flann::Matrix<float> descM1(new float[cFC.vSift_desc.size() * 128], cFC.vSift_desc.size(), 128);

	for (int iDesc = 0; iDesc < cFC.vSift_desc.size(); iDesc++)
		for (int iDim = 0; iDim < 128; iDim++)
			descM1[iDesc][iDim] = (float)cFC.vSift_desc[iDesc].vDesc[iDim];

	for (int iFeature = 0; iFeature < cFC.vSift_desc.size(); iFeature++) {
		Feature fs;
		fs.vCamera.push_back(cFC.cameraID);
		fs.vFrame.push_back(cFC.frameIdx);
		fs.vx.push_back(cFC.vSift_desc[iFeature].x);
		fs.vy.push_back(cFC.vSift_desc[iFeature].y);
		fs.vx_dis.push_back(cFC.vSift_desc[iFeature].dis_x);
		fs.vy_dis.push_back(cFC.vSift_desc[iFeature].dis_y);
		fs.vvDesc.push_back(cFC.vSift_desc[iFeature].vDesc);
		CvScalar s;
		// s = cvGet2D(iplImg1, ((int)cFC.vSift_desc[iFeature].dis_y), ((int)cFC.vSift_desc[iFeature].dis_x));
		// fs.b = s.val[0];
		// fs.g = s.val[1];
		// fs.r = s.val[2];
		feature_static.push_back(fs);
	}
	// cvReleaseImage(&iplImg1);

	flann::Index<flann::L2< float> > index1(descM1, flann::KDTreeIndexParams(4));
	index1.buildIndex();

	vector<Point> featureSequence;
	for (int iSecondFrame = currentFC + 1; iSecondFrame < vFC.size(); iSecondFrame++) {
		vector<int> vIdx1, vIdx2;
		int nn = 2;
		int nPoint1 = cFC.vSift_desc.size();
		int nPoint2 = vFC[iSecondFrame].vSift_desc.size();

		flann::Matrix<int> result12(new int[nPoint1 * nn], nPoint1, nn);
		flann::Matrix<float> dist12(new float[nPoint1 * nn], nPoint1, nn);

		flann::Matrix<int> result21(new int[nPoint2 * nn], nPoint2, nn);
		flann::Matrix<float> dist21(new float[nPoint2 * nn], nPoint2, nn);

		flann::Matrix<float> descM2(new float[vFC[iSecondFrame].vSift_desc.size() * 128], vFC[iSecondFrame].vSift_desc.size(), 128);
		for (int iDesc = 0; iDesc < vFC[iSecondFrame].vSift_desc.size(); iDesc++)
			for (int iDim = 0; iDim < 128; iDim++)
				descM2[iDesc][iDim] = (float)vFC[iSecondFrame].vSift_desc[iDesc].vDesc[iDim];

		flann::Index<flann::L2<float> > index2(descM2, flann::KDTreeIndexParams(4));
		index2.buildIndex();

		index2.knnSearch(descM1, result12, dist12, nn, flann::SearchParams(128));
		index1.knnSearch(descM2, result21, dist21, nn, flann::SearchParams(128));
		delete[] descM2.ptr();

		for (int iFeature = 0; iFeature < nPoint1; iFeature++) {
			float dist1 = dist12[iFeature][0];
			float dist2 = dist12[iFeature][1];

			if (dist1 / dist2 < 0.7) {
				int idx12 = result12[iFeature][0];

				dist1 = dist21[idx12][0];
				dist2 = dist21[idx12][1];

				if (dist1 / dist2 < 0.7) {
					int idx21 = result21[idx12][0];
					if (iFeature == idx21) {
						vIdx1.push_back(idx21);
						vIdx2.push_back(idx12);
					}
				}

				int idx21 = result21[idx12][0];
				if (iFeature == idx21) {
					vIdx1.push_back(idx21);
					vIdx2.push_back(idx12);
				}

				vIdx1.push_back(iFeature);
				vIdx2.push_back(idx12);
			}
		}

		delete[] result12.ptr();
		delete[] result21.ptr();
		delete[] dist12.ptr();
		delete[] dist21.ptr();

		vector<Point> x1, x2;
		for (int iIdx = 0; iIdx < vIdx1.size(); iIdx++) {
			Point p1, p2;
			p1.x = cFC.vSift_desc[vIdx1[iIdx]].x;
			p1.y = cFC.vSift_desc[vIdx1[iIdx]].y;

			p2.x = vFC[iSecondFrame].vSift_desc[vIdx2[iIdx]].x;
			p2.y = vFC[iSecondFrame].vSift_desc[vIdx2[iIdx]].y;

			x1.push_back(p1);
			x2.push_back(p2);
		}

		if (x1.size() < 20)
			continue;
		vector<bool> vIsInlier;
		if (GetStaticCorrespondences(x1, x2, vIsInlier) < 20)
			continue;

		vector<int> vTempIdx1, vTempIdx2;
		for (int iIsInlier = 0; iIsInlier < vIsInlier.size(); iIsInlier++) {
			if (vIsInlier[iIsInlier]) {
				if (vTempIdx1.size() > 0)
					if ((vTempIdx1[vTempIdx1.size() - 1] == vIdx1[iIsInlier]) && (vTempIdx2[vTempIdx2.size() - 1] == vIdx2[iIsInlier]))
						continue;
				vTempIdx1.push_back(vIdx1[iIsInlier]);
				vTempIdx2.push_back(vIdx2[iIsInlier]);
			}
		}
		vIdx1 = vTempIdx1;
		vIdx2 = vTempIdx2;

		if (vIdx1.size() < 20)
			continue;

		for (int iInlier = 0; iInlier < vIdx2.size(); iInlier++) {
			feature_static[vIdx1[iInlier]].vCamera.push_back(vFC[iSecondFrame].cameraID);
			feature_static[vIdx1[iInlier]].vFrame.push_back(vFC[iSecondFrame].frameIdx);
			feature_static[vIdx1[iInlier]].vx.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].x);
			feature_static[vIdx1[iInlier]].vx_dis.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].dis_x);
			feature_static[vIdx1[iInlier]].vy.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].y);
			feature_static[vIdx1[iInlier]].vy_dis.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].dis_y);
			feature_static[vIdx1[iInlier]].vvDesc.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].vDesc);

		}

		if (display) {
			IplImage *iplImg2 = cvLoadImage(vFC[iSecondFrame].imageFileName.c_str());

			for (int iInlier = 0; iInlier < vIdx2.size(); iInlier++) {
				cvCircle(iplImg2, cvPoint((int)vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].dis_x, (int)vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].dis_y), 8, cvScalar(255, 0, 0), 1);
				cvLine(iplImg2, cvPoint((int)cFC.vSift_desc[vIdx1[iInlier]].dis_x, (int)cFC.vSift_desc[vIdx1[iInlier]].dis_y), cvPoint((int)vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].dis_x, (int)vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].dis_y), cvScalar(0, 0, 0), 3);
			}
			if (iplImg2->width < 1024) {
				cvShowImage("SIFT_LOWES", iplImg2);
				cvWaitKey(50);
			}else {
				double scale = (double)(iplImg2->width) / 1024;
				CvSize size = cvSize((int)(iplImg2->width / scale), (int)(iplImg2->height / scale));
				IplImage* tmpsize = cvCreateImage(size, IPL_DEPTH_8U, 3);
				cvResize(iplImg2, tmpsize, CV_INTER_LINEAR);
				cvShowImage("SIFT_LOWES", tmpsize);
				cvWaitKey(50);
				cvReleaseImage(&tmpsize);
			}
			cvReleaseImage(&iplImg2);
		}

		cout << "Frame 1 : " << cFC.cameraID << " " << cFC.frameIdx << "  Frame 2 : " << vFC[iSecondFrame].cameraID << " " << vFC[iSecondFrame].frameIdx << " /" << vIdx1.size() << endl;
	}

	vector<Feature> vTempFeature;
	for (int iFeature = 0; iFeature < feature_static.size(); iFeature++)
		if (feature_static[iFeature].vCamera.size() > 1)
			vTempFeature.push_back(feature_static[iFeature]);
	feature_static.clear();
	feature_static = vTempFeature;
	vTempFeature.clear();

	delete[] descM1.ptr();
}

int GetStaticCorrespondences(vector<Point> x1, vector<Point> x2, vector<bool> &vIsInlier)
{
	vector<int> vInlierID;
	CvMat *cx1 = cvCreateMat(x1.size(), 2, CV_32FC1);
	CvMat *cx2 = cvCreateMat(x1.size(), 2, CV_32FC1);
	for (int ix = 0; ix < x1.size(); ix++) {
		cvSetReal2D(cx1, ix, 0, x1[ix].x);
		cvSetReal2D(cx1, ix, 1, x1[ix].y);
		cvSetReal2D(cx2, ix, 0, x2[ix].x);
		cvSetReal2D(cx2, ix, 1, x2[ix].y);
	}

	vector<cv::Point2f> points1(x1.size());
	vector<cv::Point2f> points2(x2.size());
	for (int ip = 0; ip < x1.size(); ip++) {
		cv::Point2f p1, p2;
		p1.x = x1[ip].x;
		p1.y = x1[ip].y;
		p2.x = x2[ip].x;
		p2.y = x2[ip].y;
		points1[ip] = p1;
		points2[ip] = p2;
	}

	CvMat *status = cvCreateMat(1, cx1->rows, CV_8UC1);
	CvMat *F = cvCreateMat(3, 3, CV_32FC1);
	int n = cvFindFundamentalMat(cx1, cx2, F, CV_FM_LMEDS, 1, 0.99, status);
	//cv::Mat FundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_LMEDS, 3, 0.99);
	if (n != 1) {
		cvReleaseMat(&status);
		cvReleaseMat(&F);
		cvReleaseMat(&cx1);
		cvReleaseMat(&cx2);
		return 0;
	}
	int nP = 0;
	double ave = 0;
	int nInliers = 0;
	for (int i = 0; i < cx1->rows; i++) {
		if (cvGetReal2D(status, 0, i) == 1) {
			//vIsInlier.push_back(true);
			nP++;
			CvMat *xM2 = cvCreateMat(1, 3, CV_32FC1);
			CvMat *xM1 = cvCreateMat(3, 1, CV_32FC1);
			CvMat *s = cvCreateMat(1, 1, CV_32FC1);
			cvSetReal2D(xM2, 0, 0, x2[i].x);
			cvSetReal2D(xM2, 0, 1, x2[i].y);
			cvSetReal2D(xM2, 0, 2, 1);
			cvSetReal2D(xM1, 0, 0, x1[i].x);
			cvSetReal2D(xM1, 1, 0, x1[i].y);
			cvSetReal2D(xM1, 2, 0, 1);
			cvMatMul(xM2, F, xM2);
			cvMatMul(xM2, xM1, s);

			double l1 = cvGetReal2D(xM2, 0, 0);
			double l2 = cvGetReal2D(xM2, 0, 1);
			double l3 = cvGetReal2D(xM2, 0, 2);

			double dist = abs(cvGetReal2D(s, 0, 0)) / sqrt(l1 * l1 + l2 * l2);

			if (dist < 5) {
				vIsInlier.push_back(true);
				nInliers++;
			}else
				vIsInlier.push_back(false);
			ave += dist;

			cvReleaseMat(&xM2);
			cvReleaseMat(&xM1);
			cvReleaseMat(&s);
		}else
			vIsInlier.push_back(false);
	}

	cvReleaseMat(&status);
	cvReleaseMat(&F);
	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	if (ave / nP > 10)
		return 0;
	return nInliers;
}
