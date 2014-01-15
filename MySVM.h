#pragma once
#include <opencv2/ml/ml.hpp>

class MySVM :
	public CvSVM
{
public:
	MySVM(void);
	~MySVM(void);
	//获得SVM的决策函数中的alpha数组
	double * get_alpha_vector(){
		return this->decision_func->alpha;
	}
	//获得SVM的决策函数中的rho参数,即偏移量
	float get_rho(){
		return this->decision_func->rho;
	}
};
