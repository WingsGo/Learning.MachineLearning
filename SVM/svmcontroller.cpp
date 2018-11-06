#include "svmcontroller.h"

using namespace cv;

SVMController::SVMController()
{
    m_width = 512;
    m_height = 512;
    m_img = Mat::zeros(m_height, m_width, CV_8UC3);

    m_labels[0] = 1;
    m_labels[1] = -1;
    m_labels[2] = 1;
    m_labels[3] = -1;
    m_trainingData[0][0] = 501;

    m_trainingData[0][1] = 10;
    m_trainingData[1][0] = 255;
    m_trainingData[1][1] = 10;
    m_trainingData[2][0] = 501;
    m_trainingData[2][1] = 255;
    m_trainingData[3][0] = 10;
    m_trainingData[3][1] = 501;
    //m_trainingDataMat(4, 2, CV_32FC1, m_trainingData);
    //m_labelsMat(4, 1, CV_32SC1, m_labels);
    setParams(m_svmParameters);
}

void SVMController::train()
{
   m_svm.train(m_trainingDataMat, m_labelsMat, Mat(), Mat(), m_svmParameters);
   Vec3b green(0, 255, 0), blue(255, 0, 0);
   for(int i=0; i<m_img.rows; ++i) {
       for (int j=0; j<m_img.cols; ++j) {
           Mat sampleMat = (Mat_<float>(1, 2) << j, i);
           float response = m_svm.predict(sampleMat);
           if (response == 1)
               m_img.at<Vec3b>(i, j) = green;
           else if(response == -1)
               m_img.at<Vec3b>(i, j) = blue;
       }
   }
}

void SVMController::setParams(CvSVMParams &parameters)
{
    parameters.svm_type = CvSVM::C_SVC;
    parameters.kernel_type = CvSVM::LINEAR;
    parameters.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-06);
}

void SVMController::showResult()
{
    int thickness = -1;
    int lineType = 8;
    circle( m_img, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
    circle( m_img, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( m_img, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( m_img, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

    thickness = 2;
    lineType  = 8;
    int c     = m_svm.get_support_vector_count();

    for (int i = 0; i < c; ++i)
    {
        const float* v = m_svm.get_support_vector(i);
        circle( m_img,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("result.png", m_img);        // save the image

    imshow("SVM Simple Example", m_img); // show it to the user
    waitKey(0);
}
