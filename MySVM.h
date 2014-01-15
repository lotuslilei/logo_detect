#pragma once
#include <opencv2/ml/ml.hpp>

class MySVM :
	public CvSVM
{
public:
	MySVM(void);
	~MySVM(void);
	//���SVM�ľ��ߺ����е�alpha����
	double * get_alpha_vector(){
		return this->decision_func->alpha;
	}
	//���SVM�ľ��ߺ����е�rho����,��ƫ����
	float get_rho(){
		return this->decision_func->rho;
	}
};
