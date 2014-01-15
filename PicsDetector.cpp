#include "PicsDetector.h"

vector<cv::Rect> PicsDetector::DetectAPic(const cv::Mat & src_img,float scale/*=1*/ )
{
	cv::Mat img(src_img.rows*scale, src_img.cols*scale, src_img.type());
	cv::resize(src_img,img,img.size(),0,0,cv::INTER_LINEAR);

	vector<cv::Rect> resultrects;
	if (m_usegpu)
	{
		m_hogfeatureptr->detectMultiScale(img, resultrects, 0, cv::Size(8,8));
		return resultrects;
	}
	int i,j,k;
	int stridewidth = m_win_stride.width;
	int strideheight = m_win_stride.height;
	int dim = m_hogfeatureptr->m_featuresdim;
	cv::Mat predictMat(1, dim, CV_32FC1);
	float response;
	cv::Mat tmpimg;
	for (j=0; (j+m_win_size.height) < img.rows; j+=strideheight)
	{
		for (i=0; (i + m_win_size.width) < img.cols; i+=stridewidth)
		{
			cv::Rect tmprect(i, j, m_win_size.width, m_win_size.height);
			tmpimg.release();
			img(tmprect).copyTo(tmpimg);
			m_hogfeatureptr->ExtractHogFeatures_OnePass(tmpimg);
			for (k=0; k<dim; ++k)
			{
				predictMat.at<float>(0,k) = m_hogfeatureptr->m_features[k];
			}
			response = m_predictsvmptr->predict(predictMat);
			if (response == 1)
			{
				cv::Rect outrect(i/scale, j/scale, m_win_size.width/scale, m_win_size.height/scale);
				resultrects.push_back(outrect);
			}
		}
	}
	return resultrects;
}

PicsDetector::PicsDetector( HogFeature* hogfeatureptr, MySVM* svmptr, cv::Size win_size, cv::Size win_stride, bool usegpu/* = false*/)
{
	m_hogfeatureptr = hogfeatureptr;
	m_predictsvmptr = svmptr;
	m_win_size = win_size;
	m_win_stride = win_stride;
	m_usegpu = usegpu;

	int descriptorDim;
	int supportVectorNum;

	//ʹ��gpu	
	if (m_usegpu)
	{
		//����������ά������HOG�����ӵ�ά��  
		descriptorDim = m_predictsvmptr->get_var_count();
		//֧�������ĸ��� 
		supportVectorNum = m_predictsvmptr->get_support_vector_count();
		cout<<"supportVectorNum : "<<supportVectorNum<<endl<<"dim : "<< descriptorDim<<endl;
		
		//alpha���������ȵ���֧����������
		cv::Mat alphaMat = cv::Mat::zeros(1, supportVectorNum, CV_32FC1);
		//֧����������
		cv::Mat supportVectorMat = cv::Mat::zeros(supportVectorNum, descriptorDim, CV_32FC1);
		//alpha��������֧����������Ľ��
		cv::Mat resultMat = cv::Mat::zeros(1, descriptorDim, CV_32FC1);
		
			
		//��֧�����������ݸ��Ƶ�supportVectorMat������ 
		int i,j;
		for(i=0; i<supportVectorNum; i++)  
		{  
			const float * pSVData = m_predictsvmptr->get_support_vector(i);//���ص�i��֧������������ָ��  
			for(j=0; j<descriptorDim; j++)  
			{  
				//cout<<pData[j]<<" ";  
				supportVectorMat.at<float>(i,j) = pSVData[j];  
			}  
		}

		//��alpha���������ݸ��Ƶ�alphaMat��  
		double * pAlphaData = m_predictsvmptr->get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����  
		for(int i=0; i<supportVectorNum; i++)  
		{  
			alphaMat.at<float>(0,i) = pAlphaData[i];  
		}
		
		//����-(alphaMat * supportVectorMat),����ŵ�resultMat��  
		//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�  
		resultMat = -1 * alphaMat * supportVectorMat;
		
		//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����  
		vector<float> myDetector;

		//��resultMat�е����ݸ��Ƶ�����myDetector��  
		for(int i=0; i<descriptorDim; i++)  
		{  
			myDetector.push_back(resultMat.at<float>(0,i));  
		}

		//������ƫ����rho���õ������  
		myDetector.push_back(m_predictsvmptr->get_rho());
		cout<<"myDetector's size : "<<myDetector.size()<<endl;
	}
	
}

PicsDetector::~PicsDetector( void )
{
}
