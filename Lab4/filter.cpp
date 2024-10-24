#include "filter.hpp"
#include <opencv2/opencv.hpp>
#include <pthread.h>

using namespace cv;

#define R_COEFF 0.2126
#define G_COEFF 0.7152
#define B_COEFF 0.0722

//Sobel filter coefficients
int verticalFilter[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

int horizontalFilter[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

//Thread argument structure
struct ThreadArgs {
    int startRow, endRow, startCol, endCol;
    Mat *input;
    Mat *grayOutput;
    Mat *sobelOutput;
};

//Function to convert a BGR pixel to grayscale
inline unsigned char bgrToGray(const unsigned char* bgrPixel) {
    double b = (double)bgrPixel[0];
    double g = (double)bgrPixel[1];
    double r = (double)bgrPixel[2];
    return (unsigned char)(B_COEFF * b + G_COEFF * g + R_COEFF * r);
}

//Function that each thread will execute: applies grayscale and Sobel to one quadrant
void* applyFilters(void* args) {
    ThreadArgs* threadArgs = (ThreadArgs*) args;
    int startRow = threadArgs->startRow;
    int endRow = threadArgs->endRow;
    int startCol = threadArgs->startCol;
    int endCol = threadArgs->endCol;
    Mat *input = threadArgs->input;
    Mat *grayOutput = threadArgs->grayOutput;
    Mat *sobelOutput = threadArgs->sobelOutput;

    //Apply grayscale to the section
    for (int i = startRow; i < endRow; i++) {
        unsigned char *inputRow_p = input->ptr<unsigned char>(i);
        unsigned char *outputRow_p = grayOutput->ptr<unsigned char>(i);
        for (int j = startCol; j < endCol; j++) {
            outputRow_p[j] = bgrToGray(&(inputRow_p[j * input->channels()]));
        }
    }

    //Apply Sobel to the section
    for (int i = startRow + 1; i < endRow - 1; i++) { // Padding
        for (int j = startCol + 1; j < endCol - 1; j++) {
            int vGradient = 0;
            int hGradient = 0;

            //Apply Sobel filter
            for (int m = -1; m <= 1; m++) {
                for (int n = -1; n <= 1; n++) {
                    unsigned char pixelValue = grayOutput->at<unsigned char>(i + m, j + n);
                    vGradient += pixelValue * verticalFilter[m + 1][n + 1];
                    hGradient += pixelValue * horizontalFilter[m + 1][n + 1];
                }
            }

            //Calculate magnitude of gradient
            int sum = std::min(255, std::abs(vGradient) + std::abs(hGradient));
            sobelOutput->at<unsigned char>(i - 1, j - 1) = static_cast<unsigned char>(sum);
        }
    }

    return nullptr;
}

void processFrame(Mat &input, Mat &grayOutput, Mat &sobelOutput) {
    pthread_t threads[4];
    ThreadArgs threadArgs[4];

    int rowsPerSection = input.rows / 4; // Divide rows into 4 sections

    // Create 4 threads for each horizontal section
    for (int i = 0; i < 4; i++) {
        threadArgs[i].input = &input;
        threadArgs[i].grayOutput = &grayOutput;
        threadArgs[i].sobelOutput = &sobelOutput;

        threadArgs[i].startRow = i * rowsPerSection; // Start row for the section
        threadArgs[i].endRow = (i == 3) ? input.rows : (i + 1) * rowsPerSection; // Handle last section correctly

        threadArgs[i].startCol = 0; // All sections cover the full width
        threadArgs[i].endCol = input.cols; // All sections cover the full width

        pthread_create(&threads[i], nullptr, applyFilters, &threadArgs[i]);
    }

    // Join the threads to wait for processing to complete
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], nullptr);
    }
}

