#include <iostream>
#include <fstream>
#include "HogFeature.h"
#include "MySVM.h"
#include "PicsDetector.h"
#include <opencv2/ml/ml.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/gpu/gpu.hpp>

using namespace std;
#define DEFAULTSIZE cv::Size(192,48)
int num_devices=0;

//×Ö·û´®·Ö¸îº¯Êý
std::vector<std::string> split(std::string str,std::string pattern)
{
	std::string::size_type pos;
	std::vector<std::string> result;
	str+=pattern;//À©Õ¹×Ö·û´®ÒÔ·½±ã²Ù×÷
	int size=str.size();

	for(int i=0; i<size; i++)
	{
		pos=str.find(pattern,i);
		if(pos<size)
		{
			std::string s=str.substr(i,pos-i);
			result.push_back(s);
			i=pos+pattern.size()-1;
		}
	}
	return result;
}


PicsDetector* InitData(string modelpath){
	cout<<"loading model..."<<endl;
	MySVM *svmpredictptr = new MySVM();
	svmpredictptr->load(modelpath.c_str());

	HogFeature *hogfeatureptr = new HogFeature();
	PicsDetector* picsdetectorptr;
	bool usegpu = false;
	if (num_devices > 0){
		//in gpu:
		//cell_size only support 8,8
		//Assertion failed (block_size.width % cell_size.width == 0 && block_size.height % cell_size.height == 0) in HOGDescriptor
		hogfeatureptr->CreateGpuHogDescriptor(DEFAULTSIZE,cv::Size(16,16),cv::Size(8,8),cv::Size(8,8));
		usegpu = true;
	}
	else{
		hogfeatureptr->CreateHogDescriptor_OnePass(DEFAULTSIZE,cv::Size(16,16),cv::Size(8,8),cv::Size(8,8));
	}
	picsdetectorptr = new PicsDetector(hogfeatureptr, svmpredictptr, DEFAULTSIZE, cv::Size(24,16), usegpu);
	return picsdetectorptr;
}

struct GpuBuffer{
	cv::gpu::GpuMat srcimg;
};

void AnalysePics(const cv::Mat &image, vector< vector<cv::Rect> > & rects, PicsDetector *picsdetectorptr){
	rects.push_back(picsdetectorptr->DetectAPic(image,0.6));
	rects.push_back(picsdetectorptr->DetectAPic(image,0.8));
	rects.push_back(picsdetectorptr->DetectAPic(image));
	rects.push_back(picsdetectorptr->DetectAPic(image,1.2));
	rects.push_back(picsdetectorptr->DetectAPic(image,1.4));
}

int main(int argc,char *argv[])
{
	if (argc != 5)
	{
		cout<<"Usage : "<<argv[0]<< " <input_model> <imput_img_list> <output_img_dir> <output_tmp_img_dir>"<<endl;
		exit(2);
	}
	string argv1 = argv[1];
	string argv2 = argv[2];
	string argv3 = argv[3];
	string argv4 = argv[4];
	
// 	string argv1 = "e:/jz/tmp12.model";
// 	string argv2 = "E:/jz/detectpic/detectpic.txt";
// 	string argv3 = "E:/jz/result/tmp1";
// 	string argv4 = "E:/jz/result/tmp2/";


// enable gpu
	num_devices = cv::gpu::getCudaEnabledDeviceCount();
	PicsDetector *picsdetectorptr = InitData(argv1);

	return 0;
//	reading input imgs...
	cout<<"reading images..."<<endl;
	ifstream fimgsin(argv2.c_str(), ios_base::in);
	if(!fimgsin.is_open())
	{
#ifdef DEBUG_OUTPUT
		cout<<"Open File Error"<<endl;
#endif
		exit(1);
	}


//	Analyzing imgs...
	cout<<"analyzing..."<<endl;
	string filepath;
	int totalnum=0;
	int foundnum=0;
	int counter=0;
	cv::Mat image;
	cv::Mat tmpimg;
	cv::Mat tmpresultimg;
	GpuBuffer gpubuffer;
	vector< vector<cv::Rect> > rects;
	vector< cv::Rect > rectList;
	int i,j;

	char numstr[10];
	string tmpoutpath = argv4;
	char tmpfilename[15];

	cv::Mat outputimg;
	string outputfiledir = argv3;
	string outputfilepath;
	vector<string> splitedfilepath;
	int margin = 40;


	while(getline(fimgsin,filepath))
	{
		totalnum++;
		rects.clear();
		rectList.clear();
		//gpu only support CV_8UC1 || CV_8UC4
		image = cv::imread(filepath, 0);
		cv::copyMakeBorder(image, image, margin, margin, margin, margin, cv::BORDER_REPLICATE);
		AnalysePics(image, rects, picsdetectorptr);
		image.copyTo(tmpimg);
		for (i=0; i<rects.size(); ++i)
		{
			for (j=0; j<rects.at(i).size(); ++j)
			{
				rectList.push_back(rects.at(i).at(j));
				tmpresultimg.release();
				tmpimg(rects.at(i).at(j)).copyTo(tmpresultimg);
				sprintf(tmpfilename,"%d",counter++);
				cv::imwrite(tmpoutpath + tmpfilename + ".jpg", tmpresultimg);
				cv::rectangle(image,rects.at(i).at(j),cv::Scalar(0, 0, 255));
			}
		}

		cv::groupRectangles(rectList, 1, 0.8);
		if (rectList.size())
		{
			foundnum++;
		}

		for (i=0; i<rectList.size(); ++i)
		{

// 			cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
// 			mask(rectList.at(i)).setTo(cv::Scalar(255)); 
// 			cv::inpaint(image,mask,image,5,cv::INPAINT_TELEA);
			cv::rectangle(image,rectList.at(i),cv::Scalar(255, 0, 0), 3);
		}

		outputimg.release();
		image(cv::Rect(margin,margin,image.cols-margin*2,image.rows-margin*2)).copyTo(outputimg);
		
		splitedfilepath.clear();
		splitedfilepath = split(filepath, "/");
		outputfilepath = outputfiledir + "/" + splitedfilepath[splitedfilepath.size()-1];
		cv::imwrite(outputfilepath.c_str(),outputimg);
	}

	cout<<"find : "<<foundnum<<endl;
	cout<<"miss : "<<totalnum-foundnum<<endl;
	return 0;
}
