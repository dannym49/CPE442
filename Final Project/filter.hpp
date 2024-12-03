#ifndef FILTER_HPP
#define FILTER_HPP

#include <opencv2/opencv.hpp>

using namespace cv;

// // Define RGB to Grayscale coefficients (scaled for integer math)
// #define R_COEFF 54    // 0.2126 * 256 for integer scaling
// #define G_COEFF 183   // 0.7152 * 256
// #define B_COEFF 19    // 0.0722 * 256

// // Sobel filter coefficients
// extern int verticalFilter[3][3];
// extern int horizontalFilter[3][3];

// // Structure for thread arguments
// struct ThreadArgs {
//     int startRow, endRow, startCol, endCol;
//     Mat *input;
//     Mat *grayOutput;
//     Mat *sobelOutput;
// };

// Function prototypes
void neonGrayscale(const Mat& input, Mat& grayOutput, int startRow, int endRow);
void neonSobel(const Mat& grayOutput, Mat& sobelOutput, int startRow, int endRow);
void* applyFilters(void* args);
void processFrame(Mat &input, Mat &grayOutput, Mat &sobelOutput);

#endif // FILTER_HPP

//g++ main.cpp filter.cpp -o main $(pkg-config --cflags --libs opencv4) -pthread -mcpu=cortex-a72 -std=c++11 -O0