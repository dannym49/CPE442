#include <iostream>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include "filter.hpp" // Include the header for processing functions

using namespace cv;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Error: Insufficient arguments." << std::endl;
        return -1;
    }

    VideoCapture cap;
    if (!cap.open(argv[1])) {
        std::cerr << "Could not open " << argv[1] << std::endl;
        return -1;
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video capture." << std::endl;
        return -1;
    }

    const int windowWidth = 800;
    const int windowHeight = 600;

    namedWindow("Processed Video", WINDOW_NORMAL);
    resizeWindow("Processed Video", windowWidth, windowHeight);

    while (true) {
        Mat frame;
        cap.read(frame);
        if (frame.empty()) break;

        // Create matrices for grayscale and Sobel
        Mat grayFrame(frame.rows, frame.cols, CV_8UC1);
        Mat sobelFrame(frame.rows - 2, frame.cols - 2, CV_8UC1);

        // Process the frame using threads
        processFrame(frame, grayFrame, sobelFrame);

        // Display the Sobel-processed video
        imshow("Processed Video", sobelFrame);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
