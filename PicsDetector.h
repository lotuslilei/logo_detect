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
	//使用引用,HogFeature会析构两次,挂了...
	PicsDetector(HogFeature* hogfeature, MySVM* svm, cv::Size win_size, cv::Size win_stride, bool usegpu = false);
	~PicsDetector(void);

	vector<cv::Rect> DetectAPic (const cv::Mat & src ,float scale=1);

private:

	//检测窗口大小
	cv::Size m_win_size;

	//移动窗口
	cv::Size m_win_stride;

	//预测用的svm
	MySVM * m_predictsvmptr;

	//hog特征
	HogFeature * m_hogfeatureptr;

	//use gpu or not
	bool m_usegpu;
	MySVM m_mysvm;
};
