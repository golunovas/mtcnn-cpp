#include "face_detector.hpp"

int main(int argc, char** argv) {
	FaceDetector fd("./model/");
	cv::Mat img = cv::imread("test2.jpg");
	fd.detect(img, 40.f, 0.709f);
	return 0;
}