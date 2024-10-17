#include "sobel.hpp"
#include <cmath>

using namespace cv;

const int horizontalFilter[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

const int verticalFilter[3][3] = {{-1, -2, -1}, { 0,  0,  0}, { 1,  2,  1}};

void to442_sobel(Mat &input, Mat &output) {
    CV_Assert(input.type() == CV_8UC1);

    int nRows = input.rows;
    int nCols = input.cols;
    output.create(nRows - 2, nCols - 2, CV_8UC1);

    for (int i = 1; i < nRows - 1; i++) {
        for (int j = 1; j < nCols - 1; j++) {
            int vGradient = 0;
            int hGradient = 0;

            // Calculate gradients using the Sobel filters
            for (int m = -1; m <= 1; m++) {
                for (int n = -1; n <= 1; n++) {
                    uchar pixelValue = input.at<uchar>(i + m, j + n);
                    vGradient += pixelValue * verticalFilter[m + 1][n + 1];
                    hGradient += pixelValue * horizontalFilter[m + 1][n + 1];
                }
            }

            // Calculate the magnitude of the gradient
            int sum = std::min(255, std::abs(vGradient) + std::abs(hGradient));
            output.at<uchar>(i - 1, j - 1) = static_cast<uchar>(sum);
        }
    }
}
