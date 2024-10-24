#ifndef _FILTER_H_
#define _FILTER_H_

#include <opencv2/opencv.hpp>
#include <pthread.h>

using namespace cv;

//Thread argument structure
// struct ThreadArgs {
//     int startRow, endRow, startCol, endCol; //Indices for the image section
//     Mat *input;       //Pointer to the input image
//     Mat *grayOutput;  //Pointer to the output grayscale image
//     Mat *sobelOutput; //Pointer to the output Sobel image
// };

//Function to apply grayscale and Sobel filter in a threaded manner
void* applyFilters(void* args);

//Function to process a frame: splits into quadrants and processes with threads
void processFrame(Mat &input, Mat &grayOutput, Mat &sobelOutput);

#endif // _FILTER_H_
