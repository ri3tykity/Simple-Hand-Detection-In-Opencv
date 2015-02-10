#include <stdio.h>
#include "MyFunctionsnData.h"

//#pragma region trackbar with canny
//IplImage* im = cvLoadImage("c:/i.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//IplImage* s;
//int pos1 = 1, pos2 = 1;
//void chng(int pos)
//{
//	IplImage* out = doCanny(s, pos1, pos2, 3);
//	cvNamedWindow("canny", 1);
//	cvShowImage("canny", out);
//	cvWaitKey(0);
//}
//#pragma endregion

int main()
{
	//#pragma region Load Mat image
	//	//call simple image upload.
	//	//SimpleImageUploadInWindows();
	//	//showMatviaFunction.
	//	Mat a = imread("c:/i.jpg");
	//	//write invert function for an image.
	//	MatLoad(a);
	//#pragma endregion

	//#pragma region Filtering iplimage Gaussian
	//	//filtering an image
	//	IplImage * img1 = cvLoadImage("c:/i.jpg");
	//	IplImage* s = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
	//	muiFilter(img1, s);
	//#pragma endregion

	//#pragma region trackBar with canny
	//	cvNamedWindow("trackBar", 1);
	//	//chng(2);
	//	s = cvCreateImage(cvGetSize(im), IPL_DEPTH_8U, 1);
	//	s = muiFilter(im, s);
	//	cvCreateTrackbar("pos1", "trackBar", &pos1, 100, chng);
	//	cvCreateTrackbar("pos2", "trackBar", &pos2, 100, chng);
	//	//IplImage* im = cvCreateImage(Size(200,300),IPL_DEPTH_8U,3)
	//	cvWaitKey(0);
	//#pragma endregion

	//creating line , circle and rect on black img.
	//myDraw();

	//capture from webcam and oparate for Mat type of image.
	//captureForMat_gray_hsv();

	//capture from webcam and work for iplimage.
	//webcamToIplImg();

	//machine learning...
	//machineLearningCall();
	//cvWaitKey(0);

	//Mat h1 = background_subtraction();
	
	//capture from video and store in Mat and find hand gesture.
	//handDetect();

	//Mat h2 = imread("G:/June/Programming_Google/Opencv/MyProject/hands/five.png");
	//Mat h1 = skinDetect(h2);
	//FindContor(h1);
	handDetect();

	//SVMtuts();
	//svmSimple();
	//faceDetectwSkin();

	//handDetectSkin();

	

	return 0;
}
