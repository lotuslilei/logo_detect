#include "HogFeature.h"
HogFeature::HogFeature()
{
	//构造函数
	Init();
}

HogFeature::~HogFeature()
{
	//析构函数
	if(m_cpuhog)
		delete m_cpuhog;
}

void HogFeature::Init()
{
	m_cpuhog=NULL;
	m_featuresdim=36;
	m_maxfeature = 0;
	m_minfeature = 1;
}

void HogFeature::Release()
{
	if(m_vec_cpuhog.size()>0)
	{
		for(int i=0;i<m_vec_cpuhog.size();i++)
		{
			delete m_vec_cpuhog[i];
		}
		vector<cv::HOGDescriptor *>().swap(m_vec_cpuhog);
	}
}

void HogFeature::CreateHogDescriptor(int width,int height)
{
	Release();

	for(int w=8;w<(int)(width*2/3);w+=4)
		for(int h=8;h<(int)(height*2/3);h+=4)
		{
			cv::HOGDescriptor *tmphog=new cv::HOGDescriptor(cv::Size(w,h),cv::Size(w,h),cv::Size(w/2,h/2),cv::Size(w/2,h/2),9);
			m_vec_cpuhog.push_back(tmphog);
		}
}

void HogFeature::CreateHogDescriptor_OnePass(cv::Size win_size,cv::Size block_size,cv::Size block_stride,cv::Size cell_size,int nbins)
{
	if(m_cpuhog)
	{
		delete m_cpuhog;
		m_cpuhog=NULL;
	}
	m_block_stride = block_stride;
// 	hogfeature.CreateHogDescriptor_OnePass(cv::Size(200,50),cv::Size(20,20),cv::Size(10,10),cv::Size(10,10));
// 	((200-20)/10+1)*((50-20)/10+1)*4*9 = 2736
// 	hogfeature.CreateHogDescriptor_OnePass(cv::Size(192,48),cv::Size(16,16),cv::Size(8,8),cv::Size(8,8));
// 	((192-16)/8+1)*((48-16)/8+1)*4*9 = 4140
	assert((block_size.width % cell_size.width) == 0 && (block_size.height % cell_size.height) == 0);
	assert(((win_size.width - block_size.width) % block_stride.width) == 0 && ((win_size.height - block_size.height) % block_stride.height) == 0);
	m_featuresdim = ((win_size.width - block_size.width)/block_stride.width + 1) \
					* ((win_size.height - block_size.height)/block_stride.height + 1) \
					* (block_size.width/cell_size.width) * (block_size.height/cell_size.height) \
					* nbins;
	m_cpuhog=new cv::HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins);
	m_usegpu = false;
}

void HogFeature::ExtractHogFeatures(const cv::Mat &image)
{
	m_features.clear();

	CreateHogDescriptor(image.rows,image.cols);
	for(int i=0;i<m_vec_cpuhog.size();i++)
	{
		vector<float> tmpfeatures;
		m_vec_cpuhog[i]->compute(image,tmpfeatures,m_vec_cpuhog[i]->blockStride,cv::Size(0,0));
		m_features.insert(m_features.end(),tmpfeatures.begin(),tmpfeatures.end());
	}
}

void HogFeature::ExtractHogFeatures(const cv::Mat &image,vector<float> &features)
{
	features.clear();

	ExtractHogFeatures(image);
	features=m_features;
}

void HogFeature::ExtractHogFeatures_OnePass(const cv::Mat &image,cv::Size winStride)
{
	m_features.clear();

	if(m_cpuhog)
	{
		vector<float> tmpfeatures;
		m_cpuhog->compute(image,tmpfeatures,winStride,cv::Size(0,0));
		m_featuresdim = tmpfeatures.size();
// 		for (int i=0;i<m_featuresdim;++i)
// 		{
// 			m_maxfeature = m_maxfeature > tmpfeatures[i] ? m_maxfeature : tmpfeatures[i];
// 			m_minfeature = m_minfeature < tmpfeatures[i] ? m_minfeature : tmpfeatures[i];
// 		}
		m_features.assign(tmpfeatures.begin(),tmpfeatures.end());
	}
}

void HogFeature::ExtractHogFeatures_OnePass(const cv::Mat &image,cv::Size winStride,vector<float> &features)
{
	features.clear();

	ExtractHogFeatures_OnePass(image,winStride);

	if(m_cpuhog)
	{
		features=m_features;

	}
}

void HogFeature::ExtractHogFeatures_OnePass( const cv::Mat &image )
{
	m_features.clear();
	if (m_usegpu)
	{
		if(m_gpuhog)
		{
			vector<float> tmpfeatures;
			vector<float> tmpfeatures2;
			m_featuresdim = tmpfeatures.size();
			m_features.assign(tmpfeatures.begin(),tmpfeatures.end());
		}
	}
	else{	
		if(m_cpuhog)
		{
			vector<float> tmpfeatures;
			vector<float> tmpfeatures2;
			m_cpuhog->compute(image,tmpfeatures,m_block_stride,cv::Size(0,0));
			m_featuresdim = tmpfeatures.size();
			m_features.assign(tmpfeatures.begin(),tmpfeatures.end());
			
		}
	}
}

void HogFeature::ExtractHogFeatures_OnePass(const cv::Mat &image, vector<float> &features)
{
	features.clear();

	ExtractHogFeatures_OnePass(image);

	if(m_cpuhog)
	{
		features=m_features;
	}
}

unsigned int HogFeature::GetFeaturesDim()
{
	return m_featuresdim;
}

void HogFeature::CreateGpuHogDescriptor( cv::Size win_size/*=cv::Size(64,128)*/,cv::Size block_size/*=cv::Size(16,16)*/,cv::Size block_stride/*=cv::Size(8,8)*/,cv::Size cell_size/*=cv::Size(8,8)*/,int nbins/*=9*/ )
{
	if(m_gpuhog)
	{
		delete m_gpuhog;
		m_gpuhog=NULL;
	}
	m_block_stride = block_stride;
	// 	hogfeature.CreateHogDescriptor_OnePass(cv::Size(200,50),cv::Size(20,20),cv::Size(10,10),cv::Size(10,10));
	// 	((200-20)/10+1)*((50-20)/10+1)*4*9 = 2736
	// 	hogfeature.CreateHogDescriptor_OnePass(cv::Size(192,48),cv::Size(16,16),cv::Size(8,8),cv::Size(8,8));
	// 	((192-16)/8+1)*((48-16)/8+1)*4*9 = 4140
	assert((block_size.width % cell_size.width) == 0 && (block_size.height % cell_size.height) == 0);
	assert(((win_size.width - block_size.width) % block_stride.width) == 0 && ((win_size.height - block_size.height) % block_stride.height) == 0);
	m_featuresdim = ((win_size.width - block_size.width)/block_stride.width + 1) \
		* ((win_size.height - block_size.height)/block_stride.height + 1) \
		* (block_size.width/cell_size.width) * (block_size.height/cell_size.height) \
		* nbins;
	
	//Assertion failed (block_size.width % cell_size.width == 0 && block_size.height % cell_size.height == 0) in HOGDescriptor
	//Assertion failed ((win_size.width - block_size.width ) % block_stride.width == 0 && (win_size.height - block_size.height) % block_stride.height == 0)
	//Cell size. Only (8, 8) is supported for now.
	//Block size in pixels. Align to cell size. Only (16,16) is supported for now.
	//Number of bins. Only 9 bins per cell are supported for now.
	m_gpuhog = new cv::gpu::HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins);
	m_usegpu = true;
}

void HogFeature::setSVMDetector( vector<float> myDetector )
{
	if (m_usegpu)
	{
		m_gpuhog->setSVMDetector(myDetector);
	}
	else{
		m_cpuhog->setSVMDetector(myDetector);
	}
}

void HogFeature::detectMultiScale( const cv::Mat& img, vector<cv::Rect>& found_locations, double hit_threshold/*=0*/, cv::Size win_stride/*=cv::Size()*/, cv::Size padding/*=cv::Size()*/, double scale0/*=1.05*/, int group_threshold/*=2*/ )
{
	if (m_usegpu)
	{
		m_tmpgpumat.upload(img);
		m_gpuhog->detectMultiScale(m_tmpgpumat, found_locations, hit_threshold, win_stride, padding, scale0, group_threshold);
	}
}
