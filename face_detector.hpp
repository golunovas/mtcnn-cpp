#ifndef _FACE_DETECTOR_HPP_
#define _FACE_DETECTOR_HPP_

#include <string>

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

const int NUM_REGRESSIONS = 4;
const int NUM_PTS = 5;

struct BBox {
	float x1;
	float y1;
	float x2;
	float y2;
	cv::Rect getRect() const;
	BBox getSquare() const;
};

struct Face {
	BBox bbox;
	float regression[NUM_REGRESSIONS];
	float score;
	float ptsCoords[2 * NUM_PTS];
	
	static void applyRegression(std::vector<Face>& faces);
	static void bboxes2Squares(std::vector<Face>& faces);
};

class FaceDetector {
private:
	boost::shared_ptr< caffe::Net<float> > pNet_;
    boost::shared_ptr< caffe::Net<float> > rNet_;
	boost::shared_ptr< caffe::Net<float> > oNet_;
	void initNetInput(boost::shared_ptr< caffe::Net<float> > net, cv::Mat img);
	std::vector<Face> step1(cv::Mat img, float minFaceSize, float scaleFactor);
	std::vector<Face> step2(cv::Mat img, const std::vector<Face>& faces);
	std::vector<Face> step3(cv::Mat img, const std::vector<Face>& faces);
	static std::vector<Face> nonMaximumSuppression(std::vector<Face> faces, float threshold, bool useMin = false);
	static std::vector<Face> composeFaces(const caffe::Blob<float>* regressionsBlob, 
										   const caffe::Blob<float>* scoresBlob,
										   float scale);
public:
	FaceDetector(const std::string& modelDir);
	std::vector<Face> detect(cv::Mat img, float minFaceSize, float scaleFactor);
};

#endif // _FACE_DETECTOR_HPP_