
#include "ImageProcessing.h"

int main() {

    Mat img = imread("test_bath.PNG");

    imshow("img", img);
    waitKey(0);


    Mat resultImage = EdgeDetector(img);
    imshow("img", resultImage);
    waitKey(0);
    imwrite("edge_result.png", resultImage);

    return 0;
}
