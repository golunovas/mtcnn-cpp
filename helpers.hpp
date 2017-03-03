#ifndef _HELPERS_HPP_
#define _HELPERS_HPP_

#include <opencv2/opencv.hpp>

inline cv::Mat cropAndResizeImage(cv::Mat img, cv::Rect r, cv::Size size) {
	// TODO: Remove redundant vars
	int rx = r.x;
	int ry = r.y;
	int rw = r.width;
	int rh = r. height;
	cv::Mat m = cv::Mat::zeros(rh, rw, img.type());
	int dx = std::abs(std::min(0, rx));
	if (dx > 0) { rx = 0; }
	rw -= dx;
	int dy = std::abs(std::min(0, ry));
	if (dy > 0) { ry = 0; }	
	rh -= dy;
	int dw = std::abs(std::min(0, img.cols - 1 - (rx + rw)));
	rw -= dw;
	int dh = std::abs(std::min(0, img.rows - 1 - (ry + rh)));
	rh -= dh;
	img(cv::Range(ry, ry + rh), cv::Range(rx, rx + rw)).copyTo(m(cv::Range(dy, dy + rh), cv::Range(dx, dx + rw)));
	cv::resize(m, m, size);
	return m;
}

void drawAndShowRectangle(cv::Mat img, cv::Rect r) {
	// TODO check type
	cv::Mat outImg;
	img.convertTo(outImg, CV_8UC3);
	cv::rectangle(outImg, r, cv::Scalar(0,0, 255));
	cv::imshow("test", outImg);
	cv::waitKey(0);
}

#endif //_HELPERS_HPP_