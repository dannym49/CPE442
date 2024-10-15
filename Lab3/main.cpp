
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "grayscale.hpp"
#include "sobel.hpp"

using namespace cv;

int main(int argc, char *argv[]) {
    VideoCapture cap;

    // Open camera or video file based on user input
    if (argc == 1) {
        if (!cap.open(0)) {
            std::cerr << "Error: Could not open camera." << std::endl;
            return -1;
        }
    } else if (argc == 2) {
        if (!cap.open(argv[1])) {
            std::cerr << "Could not open " << argv[1] << std::endl;
            return -1;
        }
    } else {
        std::cout << "Usage: sobel [videofile]" << std::endl;
        return -1;
    }

    // Check if the video capture is opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video capture." << std::endl;
        return -1;
    }

    while (true) {
        Mat frame, grayscale, edges;
        cap.read(frame);
        
        // Check if the frame is empty
        if (frame.empty())
            break;

        // Convert to grayscale and apply Sobel filter
        toGrayscale(frame, grayscale);
        sobel(grayscale, edges);

        // Display the original frame, grayscale frame, and edges
        imshow("Video", frame);
        imshow("Video Grayscale", grayscale);
        imshow("Video Edges", edges);

        // Exit if the user presses the 'q' key
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Release the video capture and destroy all OpenCV windows
    cap.release();
    destroyAllWindows();

    return 0;
}
