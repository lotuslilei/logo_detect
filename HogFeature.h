#ifndef HOGFEATURE_H
#define HOGFEATURE_H
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;

class HogFeature
{
	//行为
public:
	HogFeature();
	~HogFeature();

	void CreateHogDescriptor(int image_width,int image_height);
	void CreateGpuHogDescriptor(cv::Size win_size=cv::Size(64,128),cv::Size block_size=cv::Size(16,16),cv::Size block_stride=cv::Size(8,8),cv::Size cell_size=cv::Size(8,8),int nbins=9);
	void CreateHogDescriptor_OnePass(cv::Size win_size=cv::Size(64,128),cv::Size block_size=cv::Size(16,16),cv::Size block_stride=cv::Size(8,8),cv::Size cell_size=cv::Size(8,8),int nbins=9);

	void ExtractHogFeatures(const cv::Mat &image);
	void ExtractHogFeatures(const cv::Mat &image,vector<float> &features);
	void ExtractHogFeatures_OnePass(const cv::Mat &image,cv::Size winStride);
	void ExtractHogFeatures_OnePass(const cv::Mat &image);
	void ExtractHogFeatures_OnePass(const cv::Mat &image,cv::Size winStride,vector<float> &features);
	void ExtractHogFeatures_OnePass(const cv::Mat &image, vector<float> &features);
	void setSVMDetector(vector<float> myDetector);
	void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& found_locations, double hit_threshold=0, cv::Size win_stride=cv::Size(), cv::Size padding=cv::Size(), double scale0=1.05, int group_threshold=2);
	unsigned int GetFeaturesDim();

	//行为
private:
	void Init();
	void Release();
	
	//属性
public:
	vector<float> m_features;
	unsigned int m_featuresdim;
	float m_minfeature;
	float m_maxfeature;
	bool m_usegpu;

	//属性
private:
	cv::HOGDescriptor *m_cpuhog;
	cv::gpu::HOGDescriptor *m_gpuhog;
	vector<cv::HOGDescriptor *> m_vec_cpuhog;
	cv::Size m_block_stride;
	cv::gpu::GpuMat m_tmpgpumat;
};

#endif