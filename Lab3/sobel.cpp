#include "sobel.hpp"
#include <cmath>
using namespace cv;
using namespace std;

const int vertFilter[3][3] = {{-1, 0, 1}, {-2,0,2},{-1,0,1}};

const int horizFilter[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};

void to442_sobel(Mat &input, Mat &output){
    CV_Assert(input.type, )
}