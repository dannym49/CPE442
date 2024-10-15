#include "grayscale.hpp"
#include <highgui.h>


using namespace CV;

#define R_COEFF 0.2126
#define G_COEFF 0.7152
#define B_COEFF 0.0722

inline unsigned char bgrToGray(const unsigned char* bgrPixel){
    double b = (double)bgrPixel[0];
    double g = (double)bgrPixel[1];
    double r = (double)bgrPixel[2];

    return (unsigned char)(B_COEFF * b + G_COEFF * g + R_COEFF * r);
}

void to_442grayscale(Mat &input, Mat &output){
    CV_Assert(input.type() == CV_8UC3);

    int channels = input.channels();
    int nRows = input.rows;
    int ncols = input.cols;
    output.create(nRows, nCols, CV_8UC1);

    if (input.isContinuous() && output.isContinuous()){
        nCols *= nRows;
        nRows = 1;
    }

    for(int i = 0; i < nRows; i++){
        unsigned char *inputRow_p = input.ptr(i);
        unsigned char *outputRow_p = output.ptr(i);
        for(int j = 0; j < nCols; j++){
            outputRow_p[j] = bgrToGray(&(inputRow_p[j * channels]));
        }

    }
}