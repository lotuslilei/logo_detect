#pragma once
#include <iostream>
#include <fstream>
#include "HogFeature.h"
#include "MySVM.h"
#include <opencv2/ml/ml.hpp>
using namespace std;
class PicsDetector
{
public:
	//ʹ������,HogFeature����������,����...
	PicsDetector(HogFeature* hogfeature, MySVM* svm, cv::Size win_size, cv::Size win_stride, bool usegpu = false);
	~PicsDetector(void);

	vector<cv::Rect> DetectAPic (const cv::Mat & src ,float scale=1);

private:

	//��ⴰ�ڴ�С
	cv::Size m_win_size;

	//�ƶ�����
	cv::Size m_win_stride;

	//Ԥ���õ�svm
	MySVM * m_predictsvmptr;

	//hog����
	HogFeature * m_hogfeatureptr;

	//use gpu or not
	bool m_usegpu;
	MySVM m_mysvm;
};
