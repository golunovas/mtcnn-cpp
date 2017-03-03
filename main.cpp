#include "face_detector.hpp"
#include "helpers.hpp"

int main(int argc, char** argv) {
	mtcnn::FaceDetector fd("./model/", true, 0);
	cv::Mat img = cv::imread("test.jpg");
	std::vector<mtcnn::Face> faces = fd.detect(img, 40.f, 0.709f);
	for (size_t i = 0; i < faces.size(); ++i) {
		std::vector<cv::Point> pts;
		for (int p = 0; p < mtcnn::NUM_PTS; ++p) {
			pts.push_back(cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
		}
		drawAndShowFace(img, faces[i].bbox.getRect(), pts);
	}
	return 0;
}