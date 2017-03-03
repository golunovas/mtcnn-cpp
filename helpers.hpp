#ifndef _HELPERS_HPP_
#define _HELPERS_HPP_

#include <opencv2/opencv.hpp>

inline cv::Mat cropImage(cv::Mat img, cv::Rect r) {
	cv::Mat m = cv::Mat::zeros(r.height, r.width, img.type());
	int dx = std::abs(std::min(0, r.x));
	if (dx > 0) { r.x = 0; }
	r.width -= dx;
	int dy = std::abs(std::min(0, r.y));
	if (dy > 0) { r.y = 0; }	
	r.height -= dy;
	int dw = std::abs(std::min(0, img.cols - 1 - (r.x + r.width)));
	r.width -= dw;
	int dh = std::abs(std::min(0, img.rows - 1 - (r.y + r.height)));
	r.height -= dh;
	img(r).copyTo(m(cv::Range(dy, dy + r.height), cv::Range(dx, dx + r.width)));
	return m;
}

inline void drawAndShowRectangle(cv::Mat img, cv::Rect r) {
	// TODO check type
	cv::Mat outImg;
	img.convertTo(outImg, CV_8UC3);
	cv::rectangle(outImg, r, cv::Scalar(0,0, 255));
	cv::imshow("test", outImg);
	cv::waitKey(0);
}

#endif //_HELPERS_HPP_