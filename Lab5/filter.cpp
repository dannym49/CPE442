#include "filter.hpp"
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <arm_neon.h>

using namespace cv;

#define R_COEFF 54    //Equivalent of 0.2126 * 256 for NEON calculation
#define G_COEFF 183   //Equivalent of 0.7152 * 256
#define B_COEFF 19    //Equivalent of 0.0722 * 256

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

//Function to convert BGR to grayscale using NEON intrinsics
void neonGrayscale(const Mat& input, Mat& grayOutput, int startRow, int endRow) {
    const int width = input.cols * 3;  //Account for BGR channels
    
    for (int i = startRow; i < endRow; i++) {
        const uint8_t* inputRow = input.ptr<uint8_t>(i);
        uint8_t* outputRow = grayOutput.ptr<uint8_t>(i);

        for (int j = 0; j < width; j += 24) {  //Process 8 pixelst a time
            uint8x8x3_t bgr = vld3_u8(inputRow + j);  //Load 8 BGR pixels
            
            //Convert to uint16 for calculations
            uint16x8_t b = vmovl_u8(bgr.val[0]);
            uint16x8_t g = vmovl_u8(bgr.val[1]);
            uint16x8_t r = vmovl_u8(bgr.val[2]);

            //Multiply by grayscale coefficients 
            uint16x8_t gray16 = vmulq_n_u16(b, B_COEFF);
            gray16 = vmlaq_n_u16(gray16, g, G_COEFF);
            gray16 = vmlaq_n_u16(gray16, r, R_COEFF);

            //Shift right by 8 to return to 8-bit range 
            uint8x8_t gray8 = vshrn_n_u16(gray16, 8);

            //Store the grayscale pixels
            vst1_u8(outputRow + j / 3, gray8);
        }
    }
}

//Function to apply Sobel filter using NEON intrinsics
void neonSobel(const cv::Mat& grayOutput, cv::Mat& sobelOutput, int startRow, int endRow) {
    for (int i = startRow + 1; i < endRow - 1; i++) {  //Padding
        uchar* sobelRow = sobelOutput.ptr<uchar>(i - 1);

        int j = 1;
        for (; j <= grayOutput.cols - 8; j += 8) {
            int16x8_t vGradient = vdupq_n_s16(0);
            int16x8_t hGradient = vdupq_n_s16(0);

            for (int m = -1; m <= 1; m++) {
                const uchar* rowPtr = grayOutput.ptr<uchar>(i + m);

                //Load 8 pixels from the row 
                uint8x8_t pixels_u8_0 = vld1_u8(&rowPtr[j - 1]);
                uint8x8_t pixels_u8_1 = vld1_u8(&rowPtr[j]);
                uint8x8_t pixels_u8_2 = vld1_u8(&rowPtr[j + 1]);

                //Convert to signed 16 bit integers
                int16x8_t pixels_0 = vreinterpretq_s16_u16(vmovl_u8(pixels_u8_0));
                int16x8_t pixels_1 = vreinterpretq_s16_u16(vmovl_u8(pixels_u8_1));
                int16x8_t pixels_2 = vreinterpretq_s16_u16(vmovl_u8(pixels_u8_2));

                //Apply vertical filter for each pixel col
                vGradient = vmlaq_n_s16(vGradient, pixels_0, verticalFilter[m + 1][0]);
                vGradient = vmlaq_n_s16(vGradient, pixels_1, verticalFilter[m + 1][1]);
                vGradient = vmlaq_n_s16(vGradient, pixels_2, verticalFilter[m + 1][2]);

                //Apply horizontal filter for each pixel col
                hGradient = vmlaq_n_s16(hGradient, pixels_0, horizontalFilter[m + 1][0]);
                hGradient = vmlaq_n_s16(hGradient, pixels_1, horizontalFilter[m + 1][1]);
                hGradient = vmlaq_n_s16(hGradient, pixels_2, horizontalFilter[m + 1][2]);
            }

            //Calculate magnitude of gradient
            int16x8_t gradient = vqaddq_s16(vabsq_s16(vGradient), vabsq_s16(hGradient));
            uint8x8_t result = vqmovun_s16(gradient);

            vst1_u8(&sobelRow[j - 1], result);  //Store result
        }

        //Handle remaining pixels without NEON 
        for (; j < grayOutput.cols - 1; j++) {
            int vGradient = 0;
            int hGradient = 0;

            for (int m = -1; m <= 1; m++) {
                for (int n = -1; n <= 1; n++) {
                    unsigned char pixelValue = grayOutput.at<unsigned char>(i + m, j + n);
                    vGradient += pixelValue * verticalFilter[m + 1][n + 1];
                    hGradient += pixelValue * horizontalFilter[m + 1][n + 1];
                }
            }

            int sum = std::min(255, std::abs(vGradient) + std::abs(hGradient));
            sobelOutput.at<unsigned char>(i - 1, j - 1) = static_cast<unsigned char>(sum);
        }
    }
}

//Function to apply grayscale and sobel to section
void* applyFilters(void* args) {
    ThreadArgs* threadArgs = (ThreadArgs*) args;
    int startRow = threadArgs->startRow;
    int endRow = threadArgs->endRow;
    Mat *input = threadArgs->input;
    Mat *grayOutput = threadArgs->grayOutput;
    Mat *sobelOutput = threadArgs->sobelOutput;

    //Apply grayscale 
    neonGrayscale(*input, *grayOutput, startRow, endRow);

    //Apply Sobel filter 
    neonSobel(*grayOutput, *sobelOutput, startRow, endRow);

    return nullptr;
}

//Main processing function to create threads and apply filters
void processFrame(Mat &input, Mat &grayOutput, Mat &sobelOutput) {
    pthread_t threads[4];
    ThreadArgs threadArgs[4];

    int rowsPerSection = input.rows / 4;

    for (int i = 0; i < 4; i++) {
        threadArgs[i].input = &input;
        threadArgs[i].grayOutput = &grayOutput;
        threadArgs[i].sobelOutput = &sobelOutput;

        threadArgs[i].startRow = i * rowsPerSection;
        threadArgs[i].endRow = (i == 3) ? input.rows : (i + 1) * rowsPerSection;

        pthread_create(&threads[i], nullptr, applyFilters, &threadArgs[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], nullptr);
    }
}
