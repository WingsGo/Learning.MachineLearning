#ifndef SVMCONTROLLER_H
#define SVMCONTROLLER_H

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>

class SVMController
{
public:
    SVMController();
    void train();
    void showResult();

private:
    void setParams(CvSVMParams& parameters);

private:
    int m_width;
    int m_height;
    int m_labels[4];
    float m_trainingData[4][2];
    cv::Mat m_img;
    cv::Mat m_trainingDataMat;
    cv::Mat m_labelsMat;
    CvSVMParams m_svmParameters;
    CvSVM m_svm;
};

#endif // SVMCONTROLLER_H
