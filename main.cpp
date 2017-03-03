#include <iostream>
#include "face_detector.hpp"
#include "helpers.hpp"

int main(int argc, char** argv) {
	Timer timer;
	mtcnn::FaceDetector fd("./model/", 0.6f, 0.7f, 0.7f, true, 0);
	cv::Mat img = cv::imread("test.jpg");
	timer.start();
	std::vector<mtcnn::Face> faces = fd.detect(img, 40.f, 0.709f);
	std::cout << "Elapsed time (ms): " << timer.stop() << std::endl;
	for (size_t i = 0; i < faces.size(); ++i) {
		std::vector<cv::Point> pts;
		for (int p = 0; p < mtcnn::NUM_PTS; ++p) {
			pts.push_back(cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
		}
		drawAndShowFace(img, faces[i].bbox.getRect(), pts);
	}
	return 0;
}