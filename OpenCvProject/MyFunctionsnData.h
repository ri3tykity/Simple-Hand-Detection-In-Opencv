#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include <opencv\ml.h>
#include <iostream>
#include <math.h>
#include <string>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>

using namespace cv;
using namespace std;

bool plotSupportVectors = true;
int numTrainingPoints = 200;
int numTestPoints = 2000;
int size = 200;
int eq = 0;

//Global variable experimental image.
IplImage* iplImg = cvLoadImage("c:/i.jpg");
Mat matImg = imread("c:/colored_q.png");

//SIMPLE IMAGE DISPLAY FUNCTION.
int SimpleImageUploadInWindows()
{
	//create 200x300 black image.
	cv::Mat img(200, 300, CV_32SC3);
	namedWindow("MyCreatedImg", CV_WINDOW_AUTOSIZE);
	imshow("MyCreatedImg", img);
	//load image from c drive
	IplImage * img1 = cvLoadImage("c:/i.jpg");
	cvNamedWindow("MyLoadImg", CV_WINDOW_AUTOSIZE);
	cvShowImage("MyLoadImg", img1);
	//destroy created windows n memory when user press escape-key.
	waitKey(0);
	cvReleaseImage(&img1);
	cvDestroyWindow("MyLoadImg");

	return 0;
}

//TEST WITH MAT ARGUMENT
int MatLoad(Mat& a)
{
	namedWindow("LoadMat", CV_WINDOW_AUTOSIZE);
	imshow("LoadMat", a);
	waitKey(0);

	return 0;
}

//Gaussian filter iplimg
IplImage* muiFilter(IplImage* img1, IplImage* s)
{
	cvSmooth(img1, s, CV_GAUSSIAN, 3, 3);
	//cvNamedWindow("MyLoadImg", CV_WINDOW_AUTOSIZE);
	//cvShowImage("MyLoadImg", img1);
	//cvNamedWindow("filteredImg", CV_WINDOW_AUTOSIZE);
	//cvShowImage("filteredImg", s);
	//cvWaitKey(0);
	return s;
}

IplImage* doCanny(IplImage* in, double lowThress, double highThress, double aperture)
{
	if (in->nChannels != 1)
	{
		printf("error in image");
		return 0;
	}
	IplImage* out = cvCreateImage(cvGetSize(in), IPL_DEPTH_8U, 1);
	cvCanny(in, out, lowThress, highThress, aperture);
	return out;
}

void myDraw()
{
	Mat imgM = Mat::zeros(300, 200, CV_8UC3);
	//Mat imgM = imread("c:/i.jpg");
	line(imgM, Point(0, 150), Point(200, 150), Scalar(0, 255, 0), 1);
	circle(imgM, Point(100, 75), 50, Scalar(10, 255, 255), -1);
	rectangle(imgM, Point(50, 170), Point(100, 250), Scalar(0, 0, 255), 3);

	cvNamedWindow("DRAWW", CV_WINDOW_AUTOSIZE);
	imshow("DRAWW", imgM);
	cvWaitKey(0);
}

void captureForMat_gray_hsv()
{
	VideoCapture capture(0); //-1, 0, 1 device id
	if (!capture.isOpened())
	{
		printf("error to initialize camera");
		exit(0);
	}
	Mat matCapture, gray_img, hsv_img;
	cv::Mat prev_img(Size(640, 480), IPL_DEPTH_8U, 3);
	cv::Mat out(Size(640, 480), IPL_DEPTH_8U, 1);
	while (1)
	{
		capture >> matCapture;
		cvtColor(matCapture, matCapture, CV_BGR2GRAY);
		//cvtColor(matCapture, hsv_img, CV_BGR2HSV);
		//subtract(matCapture, prev_img, out);
		imshow("original", matCapture);
		imshow("sub", out);
		//imshow("hsv", hsv_img);
		char c = waitKey(1);
		if (c == 27)
			break;
	}
	cvDestroyAllWindows();
}

void webcamToIplImg()
{
	CvCapture* capture_webcam;
	capture_webcam = cvCaptureFromCAM(0);
	IplImage* orig;
	IplImage* res;
	res = cvCreateImage(Size(640, 480), IPL_DEPTH_8U, 3);
	if (capture_webcam == NULL)
	{
		printf("error capture from webcam");
		exit(0);
	}
	while (true)
	{
		orig = cvQueryFrame(capture_webcam);
		//cvSmooth(orig, orig, CV_GAUSSIAN, 9, 9);
		//cvInRangeS(orig, CV_RGB(0, 0, 0), CV_RGB(250, 200, 200), res);
		if (orig == NULL)
			break;
		cvSub(orig, res, res);
		//res = orig;
		cvShowImage("orig", orig);
		cvShowImage("res", res);
		char c = cvWaitKey(1);
		if (c == 27)
			break;
	}
	destroyAllWindows();
}

Mat background_subtraction()
{
	Mat h1 = imread("G:/June/Programming_Google/Opencv/MyProject/hands/bg.png");
	Mat h2 = imread("G:/June/Programming_Google/Opencv/MyProject/hands/five.png");
	Mat bgsub;
	absdiff(h1, h2, bgsub);
	//threshold(bgsub, bgsub, 45, 255, CV_THRESH_BINARY);
	//skinDetect(bgsub);
	imshow("diff", bgsub);
	return bgsub;
}

Mat findContor(Mat img, Mat b)
{
	//resize(img, img, Size(640, 480), 0, 0, INTER_CUBIC);
	Mat tress_out;
	vector<vector<Point>> counter;
	vector<Vec4i> hie;

	threshold(img, tress_out, 50, 255, THRESH_BINARY);
#pragma region For face detection
	//for face detection
	//Mat ele = getStructuringElement(MORPH_ELLIPSE, Size(20, 20), Point(0, 0));
	//dilate(tress_out, tress_out, ele);
	//imshow("ele", tress_out);
#pragma endregion
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* HandContour = NULL;
	CvSeq* Fingers;
	//find contours
	findContours(tress_out, counter, hie, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//cvFindContours()
	vector<Moments> mu(counter.size());
	//find covex hull
	vector<vector<Point>> hull(counter.size());
	for(int i = 0; i < counter.size(); i++)
	{
		double siz = contourArea(counter[i], false);
		//printf("\nhand size is : %f", siz);
		if (siz>4000)
		{
			printf("Contour area : %f\n", siz);
			//printf("\n in sixe");
			mu[i] = moments(counter[i], false);
			convexHull(Mat(counter[i]), hull[i], false);
		}
		else
		{

		}
	}
	vector<Point2f> mc(counter.size());
	for (int i = 0; i < counter.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		drawContours(b, hull, i, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
		circle(b, mc[i], 3, Scalar(0, 0, 255), -1, 8, 0);
		circle(b, mc[i], 5, Scalar(0, 255, 255), 1, 8, 0);
	}
	//imshow("hull", draw);
	//cvWaitKey(1);
	return b;
}

Mat xactHand(Mat a, Mat b)
{
	Mat tmp(a.size(), a.type());
	uchar det;
	for (int j = 0; j < a.rows; j++)
	{
		for (int k = 0; k < a.cols; k++)
		{
			det = b.at<uchar>(j, k);
			for (int i = 0; i < a.channels(); i++)
			{
				if (det > 100)
				{
					tmp.at<Vec3b>(j, k)[i] = a.at<Vec3b>(j, k)[i];
				}
				else
				{
					tmp.at<Vec3b>(j, k)[i] = 0;
				}
			}
		}
	}
	return tmp;
}

Mat skinDetect(Mat h1)
{
	//Mat h1 = imread("G:/June/Programming_Google/Opencv/MyProject/h1.jpg");
	Mat skin(Size(h1.rows, h1.cols), IPL_DEPTH_8U, 3);
	//Mat oyt(Size(h1.rows, h1.cols), IPL_DEPTH_8U, 1);
	//imshow("orig", h1);
	cvtColor(h1, skin, CV_BGR2YCrCb);
	//cvtColor(h1, oyt, CV_BGR2GRAY);
	inRange(skin, Scalar(0, 133, 77), Scalar(255, 173, 127), skin);
	//inRange(skin, Scalar(0, 50, 10), Scalar(255, 200, 200), skin);
	//Mat xatHand = xactHand(h1,skin);
	imshow("skin", skin);
	//cvWaitKey(0);
	return skin;
}

void handDetect()
{
	CvCapture* capture;
	capture = cvCaptureFromFile("hand1.mp4");
	//cap = cvCaptureFromFile("");
	if (capture == NULL)
	{
		printf("error to initialize camera");
		exit(0);
	}
	Mat tst;
	while (true)
	{
		tst = cvQueryFrame(capture);
		//resize(tst, tst, Size(480, 340));
		Mat rst = skinDetect(tst);
		Mat hu = findContor(rst, tst);
		//Mat res = add(tst, hu, hu);
		//imshow("hand", rst);
		imshow("hull", hu);
		cvWaitKey(1);
	}
}

void SVMtuts()
{
	Mat I = Mat::zeros(512, 512, CV_8UC3);
	//----------------Setup training data randomly-----------------//
	Mat trainData(200, 2, CV_32FC1);
	Mat labels(200, 1, CV_32FC1);

	RNG rng(100);  // random value generation class.

	//set up the linearly separable part of the training data
	int nLinearSamples = (int)(0.9f * 100);
	//generate sample points for class 1
	Mat trainClass = trainData.rowRange(0, nLinearSamples);
	//The x coordinate of point is in [0,0.4)
	Mat c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * 512));
	//The y coordinate of point is in [0,1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(512));

	//generate sample points for class 2
	trainClass = trainData.rowRange(200 - nLinearSamples, 200);
	//The x coordinate of point is in [0,0.4)
	c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.6 * 512), Scalar(512));
	//The y coordinate of point is in [0,1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(512));

	//-----------set of non linear set of saparable data---------------

	//generate random points for class 1 and class 2
	trainClass = trainData.rowRange(nLinearSamples, 200 - nLinearSamples);
	c = trainClass.colRange(0, 1);
	rng.fill(c, RNG::UNIFORM, Scalar(0.4 * 512), Scalar(0.6 * 512));

	c = trainClass.colRange(1, 2);
	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(512));

	//-------------set up the labels for the classes--------------------
	labels.rowRange(0, 100).setTo(1);  //class 1
	labels.rowRange(100, 200).setTo(2);//class 2

	//-------------set up the support vector machine parameter-----------
	CvSVMParams params;
	params.svm_type = SVM::C_SVC;
	params.C = 0.1;
	params.kernel_type = SVM::LINEAR;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

	//--------train the svm--------------
	cout << "Starting training process....." << endl;
	CvSVM svm;
	svm.train(trainData, labels, Mat(), Mat(), params);
	cout << "finished training process....." << endl;

	//--------show the descison region-------------------------------------

	Vec3b green(0, 100, 0), blue(100, 0, 0);
	for (int i = 0; i < I.rows; ++i)
	{
		for (int j = 0; j < I.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
			float response = svm.predict(sampleMat);

			if (response == 1) I.at<Vec3b>(j, i) = green;
			else if (response == 2) I.at<Vec3b>(j, i) = blue;

		}
	}

	//-----------show the training data---------------------------------------
	float px, py;
	//class 1
	for (int i = 0; i < 100; ++i)
	{
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(I, Point((int)px, (int)py), 3, Scalar(0, 255, 0), -1, 8);
	}
	//class 2
	for (int i = 100; i < 200; ++i)
	{
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(I, Point((int)px, (int)py), 3, Scalar(255, 0, 0), -1, 8);
	}

	//-------------show support vectors------------------------------------------
	int x = svm.get_support_vector_count();
	for (int i = 0; i < x; ++i)
	{
		const float* v = svm.get_support_vector(i);
		circle(I, Point((int)v[0], (int)v[1]), 6, Scalar(0, 255, 255), 2, 8);
	}
	//imwrite("result.png",I);
	imshow("result", I);
	waitKey(0);
}

void svmSimple()
{
	Mat  img = Mat::zeros(512, 512, CV_8UC3);
	//set up training data
	float labels[5] = { 1.0, 2.0, 3.0, 4.0, 4.0 };
	Mat labelsMat(5, 1, CV_32FC1, labels);

	float trainingData[5][2] = { { 510, 10 }, { 255, 10 }, { 300, 255 }, { 10, 501 }, { 150, 200 } };
	Mat trainingDataMat(5, 2, CV_32FC1, trainingData);

	//set up SVM's parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, 100, 1e6);

	//train the svm;
	CvSVM svm;
	svm.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255), yellow(0, 255, 255);
	//show decision region shown by svm
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = svm.predict(sampleMat);
			if (response == 1)
				img.at<Vec3b>(i, j) = green;
			else if (response == 2)
				img.at<Vec3b>(i, j) = blue;
			else if (response == 3)
				img.at<Vec3b>(i, j) = red;
			else if (response == 4)
				img.at<Vec3b>(i, j) = yellow;
		}
	}

	//show the training data
	circle(img, Point(501, 10), 5, Scalar(0, 0, 255), -1, 8);
	circle(img, Point(255, 10), 5, Scalar(0, 255, 255), -1, 8);
	circle(img, Point(501, 255), 5, Scalar(255, 255, 0), -1, 8);
	circle(img, Point(10, 501), 5, Scalar(255, 255, 255), -1, 8);
	circle(img, Point(150, 200), 5, Scalar(255, 0, 255), -1, 8);

	//show support vectors
	int c = svm.get_support_vector_count();
	for (int i = 0; i < c; ++i)
	{
		const float* v = svm.get_support_vector(i);
		circle(img, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), 2, 8);
	}

	imshow("result", img);
	waitKey(0);
}

void handDetectSkin()
{
	VideoCapture capture(0); //-1, 0, 1 device id
	if (!capture.isOpened())
	{
		printf("error to initialize camera");
		exit(0);
	}
	Mat cap_img;
	while (1)
	{
		capture >> cap_img;
		waitKey(10);
		//resize(cap_img, cap_img, Size(480, 340));
		Mat skin = skinDetect(cap_img);
		Mat res = findContor(skin, cap_img);
		//imshow("skin", skin);
		imshow("res", res);
		waitKey(1);
	}
	exit(0);
}

int faceDetectwSkin()
{
	CascadeClassifier face_cascade, eye_cascade;
	if (!face_cascade.load("c:\\haar\\haarcascade_frontalface_alt2.xml")) {
		printf("Error loading cascade file for face");
		return 1;
	}
	if (!eye_cascade.load("c:\\haar\\haarcascade_eye.xml")) {
		printf("Error loading cascade file for eye");
		return 1;
	}
	VideoCapture capture(0); //-1, 0, 1 device id
	if (!capture.isOpened())
	{
		printf("error to initialize camera");
		return 1;
	}
	Mat cap_img, gray_img;
	vector<Rect> faces, eyes;
	while (1)
	{
		capture >> cap_img;
		cvtColor(cap_img, gray_img, CV_BGR2GRAY);
		cv::equalizeHist(gray_img, gray_img);
		face_cascade.detectMultiScale(gray_img, faces, 1.1, 10, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, cvSize(0, 0), cvSize(300, 300));
		Mat res, tst;
		for (int i = 0; i < faces.size(); i++)
		{
			Point pt1(faces[i].x + faces[i].width - 15, faces[i].y + faces[i].height - 15);
			Point pt2(faces[i].x + 15, faces[i].y + 15);
			Mat faceROI = gray_img(faces[i]);
			Mat newFaceROI(cap_img, Rect(pt1, pt2));
			tst = skinDetect(newFaceROI);
			res = findContor(tst, newFaceROI);
			//rectangle(cap_img, pt1, pt2, cvScalar(0, 255, 0), 2, 8, 0);
		}
		imshow("res", tst);
		imshow("Result", cap_img);
		waitKey(3);
		char c = waitKey(3);
		if (c == 27)
			break;
	}
	return 0;

}

#pragma region Machine Learning using different methods.
// accuracy
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
	assert(predicted.rows == actual.rows);
	int t = 0;
	int f = 0;
	for (int i = 0; i < actual.rows; i++) {
		float p = predicted.at<float>(i, 0);
		float a = actual.at<float>(i, 0);
		if ((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
			t++;
		}
		else {
			f++;
		}
	}
	return (t * 1.0) / (t + f);
}
// plot data and class
void plot_binary(cv::Mat& data, cv::Mat& classes, string name) {
	cv::Mat plot(size, size, CV_8UC3);
	plot.setTo(cv::Scalar(255.0, 255.0, 255.0));
	for (int i = 0; i < data.rows; i++) {

		float x = data.at<float>(i, 0) * size;
		float y = data.at<float>(i, 1) * size;

		if (classes.at<float>(i, 0) > 0) {
			cv::circle(plot, Point(x, y), 2, CV_RGB(255, 0, 0), 1);
		}
		else {
			cv::circle(plot, Point(x, y), 2, CV_RGB(0, 255, 0), 1);
		}
	}
	cv::imshow(name, plot);
}

// function to learn
int f(float x, float y, int equation) {
	switch (equation) {
	case 0:
		return y > sin(x * 10) ? -1 : 1;
		break;
	case 1:
		return y > cos(x * 10) ? -1 : 1;
		break;
	case 2:
		return y > 2 * x ? -1 : 1;
		break;
	case 3:
		return y > tan(x * 10) ? -1 : 1;
		break;
	default:
		return y > cos(x * 10) ? -1 : 1;
	}
}

// label data with equation
cv::Mat labelData(cv::Mat points, int equation) {
	cv::Mat labels(points.rows, 1, CV_32FC1);
	for (int i = 0; i < points.rows; i++) {
		float x = points.at<float>(i, 0);
		float y = points.at<float>(i, 1);
		labels.at<float>(i, 0) = f(x, y, equation);
	}
	return labels;
}

void svm(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
	CvSVMParams param = CvSVMParams();

	param.svm_type = CvSVM::C_SVC;
	param.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
	param.degree = 0; // for poly
	param.gamma = 20; // for poly/rbf/sigmoid
	param.coef0 = 0; // for poly/sigmoid

	param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
	param.p = 0.0; // for CV_SVM_EPS_SVR

	param.class_weights = NULL; // for CV_SVM_C_SVC
	param.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-6;

	// SVM training (use train auto for OpenCV>=2.0)
	CvSVM svm(trainingData, trainingClasses, cv::Mat(), cv::Mat(), param);

	cv::Mat predicted(testClasses.rows, 1, CV_32F);

	for (int i = 0; i < testData.rows; i++) {
		cv::Mat sample = testData.row(i);

		float x = sample.at<float>(0, 0);
		float y = sample.at<float>(0, 1);

		predicted.at<float>(i, 0) = svm.predict(sample);
	}

	cout << "Accuracy_{SVM} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions SVM");

	// plot support vectors
	if (plotSupportVectors) {
		cv::Mat plot_sv(size, size, CV_8UC3);
		plot_sv.setTo(cv::Scalar(255.0, 255.0, 255.0));

		int svec_count = svm.get_support_vector_count();
		for (int vecNum = 0; vecNum < svec_count; vecNum++) {
			const float* vec = svm.get_support_vector(vecNum);
			cv::circle(plot_sv, Point(vec[0] * size, vec[1] * size), 3, CV_RGB(0, 0, 0));
		}
		cv::imshow("Support Vectors", plot_sv);
	}
}

void mlp(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

	cv::Mat layers = cv::Mat(4, 1, CV_32SC1);

	layers.row(0) = cv::Scalar(2);
	layers.row(1) = cv::Scalar(10);
	layers.row(2) = cv::Scalar(15);
	layers.row(3) = cv::Scalar(1);

	CvANN_MLP mlp;
	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	mlp.create(layers);

	// train
	mlp.train(trainingData, trainingClasses, cv::Mat(), cv::Mat(), params);

	cv::Mat response(1, 1, CV_32FC1);
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		cv::Mat response(1, 1, CV_32FC1);
		cv::Mat sample = testData.row(i);

		mlp.predict(sample, response);
		predicted.at<float>(i, 0) = response.at<float>(0, 0);

	}

	cout << "Accuracy_{MLP} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions Backpropagation");
}

void knn(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses, int K) {

	CvKNearest knn(trainingData, trainingClasses, cv::Mat(), false, K);
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const cv::Mat sample = testData.row(i);
		predicted.at<float>(i, 0) = knn.find_nearest(sample, K);
	}

	cout << "Accuracy_{KNN} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions KNN");
}

void bayes(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

	CvNormalBayesClassifier bayes(trainingData, trainingClasses);
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const cv::Mat sample = testData.row(i);
		predicted.at<float>(i, 0) = bayes.predict(sample);
	}

	cout << "Accuracy_{BAYES} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions Bayes");
}

void decisiontree(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses)
{

	CvDTree dtree;
	cv::Mat var_type(3, 1, CV_8U);

	// define attributes as numerical
	var_type.at<unsigned int>(0, 0) = CV_VAR_NUMERICAL;
	var_type.at<unsigned int>(0, 1) = CV_VAR_NUMERICAL;
	// define output node as numerical
	var_type.at<unsigned int>(0, 2) = CV_VAR_NUMERICAL;

	dtree.train(trainingData, CV_ROW_SAMPLE, trainingClasses, cv::Mat(), cv::Mat(), var_type, cv::Mat(), CvDTreeParams());
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const cv::Mat sample = testData.row(i);
		CvDTreeNode* prediction = dtree.predict(sample);
		predicted.at<float>(i, 0) = prediction->value;
	}

	cout << "Accuracy_{TREE} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions tree");
}

void machineLearningCall()
{
	cv::Mat trainingData(numTrainingPoints, 2, CV_32FC1);
	cv::Mat testData(numTestPoints, 2, CV_32FC1);

	cv::randu(trainingData, 0, 1);
	cv::randu(testData, 0, 1);

	cv::Mat trainingClasses = labelData(trainingData, eq);
	cv::Mat testClasses = labelData(testData, eq);

	plot_binary(trainingData, trainingClasses, "Training Data");
	plot_binary(testData, testClasses, "Test Data");

	svm(trainingData, trainingClasses, testData, testClasses);
	mlp(trainingData, trainingClasses, testData, testClasses);
	knn(trainingData, trainingClasses, testData, testClasses, 3);
	bayes(trainingData, trainingClasses, testData, testClasses);
	//	decisiontree(trainingData, trainingClasses, testData, testClasses);
}
#pragma endregion


