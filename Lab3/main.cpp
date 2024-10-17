#include <iostream>
#include <opencv2/opencv.hpp>

#include "grayscale.hpp"
#include "sobel.hpp"

using namespace cv;

int main(int argc, char *argv[]) {
    //Check if the number of arguments is exactly 2
    if (argc != 2) {
        std::cerr << "Error: Insufficient arguments." << std::endl;
        return -1;
    }

    VideoCapture cap;

    //Try to open the video file provided as the first argument
    if (!cap.open(argv[1])) {
        std::cerr << "Could not open " << argv[1] << std::endl;
        return -1;
    }

    //Check if the video capture is opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video capture." << std::endl;
        return -1;
    }

    //Define the desired window size
    const int windowWidth = 800; // Set your desired width
    const int windowHeight = 600; // Set your desired height

    //Create a window with the desired size
    namedWindow("Processed Video", WINDOW_NORMAL);
    resizeWindow("Processed Video", windowWidth, windowHeight);

    while (true) {
        Mat frame, grayscale, edges;
        cap.read(frame);
        
        //Check if the frame is empty
        if (frame.empty())
            break;

        //Convert to grayscale and apply Sobel filter
        to442_grayscale(frame, grayscale);
        to442_sobel(grayscale, edges);

        //Display the final processed frame with Sobel applied
        imshow("Processed Video", edges);

        //Exit if the user presses the 'q' key
        if (waitKey(1) == 'q') {
            break;
        }
    }

    //Release the video capture and destroy all OpenCV windows
    cap.release();
    destroyAllWindows();

    return 0;
}
