#include<opencv\cv.h>
#include <opencv\highgui.h>

using namespace cv;
using namespace std;

int SimpleImageUploadInWindows()
{
	cv::Mat img(333, 200, CV_32SC3);
	namedWindow("My Window", 1);
	imshow("My Window", img);
	waitKey(0);
	return 0;
}