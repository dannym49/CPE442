#ifndef FILTER_HPP
#define FILTER_HPP

#include <opencv2/opencv.hpp>

using namespace cv;



// Function prototypes
void neonGrayscale(const Mat& input, Mat& grayOutput, int startRow, int endRow);
void neonSobel(const Mat& grayOutput, Mat& sobelOutput, int startRow, int endRow);
void* applyFilters(void* args);
void processFrame(Mat &input, Mat &grayOutput, Mat &sobelOutput);

#endif // FILTER_HPP
