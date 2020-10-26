#include "MatrixReaderWriter.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void drawPoints(MatrixReaderWriter &mrw, float u, float v, float rad, Mat &resImg)
{
    int NUM = mrw.rowNum;

    Mat C(3, 3, CV_32F);
    Mat R(3, 3, CV_32F);
    Mat T(3, 1, CV_32F);

    float tx = cos(u) * sin(v);
    float ty = sin(u) * sin(v);
    float tz = cos(v);

    //Intrincic parameters

    C.at<float>(0, 0) = 3000.0f;
    C.at<float>(0, 1) = 0.0f;
    C.at<float>(0, 2) = 400.0f;

    C.at<float>(1, 0) = 0.0f;
    C.at<float>(1, 1) = 3000.0f;
    C.at<float>(1, 2) = 300.0f;

    C.at<float>(2, 0) = 0.0f;
    C.at<float>(2, 1) = 0.0f;
    C.at<float>(2, 2) = 1.0f;

    T.at<float>(0, 0) = rad * tx;
    T.at<float>(1, 0) = rad * ty;
    T.at<float>(2, 0) = rad * tz;

    //Mirror?
    int HowManyPi = (int)floor(v / 3.1415);

    //Axes:
    Point3f Z(-1.0 * tx, -1.0 * ty, -1.0 * tz);
    Point3f X(sin(u) * sin(v), -cos(u) * sin(v), 0.0f);
    if (HowManyPi % 2)
        X = (1.0 / sqrt(X.x * X.x + X.y * X.y + X.z * X.z)) * X;
    else
        X = (-1.0 / sqrt(X.x * X.x + X.y * X.y + X.z * X.z)) * X;

    Point3f up = X.cross(Z);

    /*
	printf("%f\n",X.x*X.x+X.y*X.y+X.z*X.z);
	printf("%f\n",up.x*up.x+up.y*up.y+up.z*up.z);
	printf("%f\n",Z.x*Z.x+Z.y*Z.y+Z.z*Z.z);

	printf("(%f,%f)\n",u,v);
*/

    R.at<float>(2, 0) = Z.x;
    R.at<float>(2, 1) = Z.y;
    R.at<float>(2, 2) = Z.z;

    R.at<float>(1, 0) = up.x;
    R.at<float>(1, 1) = up.y;
    R.at<float>(1, 2) = up.z;

    R.at<float>(0, 0) = X.x;
    R.at<float>(0, 1) = X.y;
    R.at<float>(0, 2) = X.z;

    for (int i = 0; i < NUM; i++)
    {
        Mat vec(3, 1, CV_32F);
        vec.at<float>(0, 0) = mrw.data[3 * i];
        vec.at<float>(1, 0) = mrw.data[3 * i + 1];
        vec.at<float>(2, 0) = mrw.data[3 * i + 2];

        int red = 255;
        int green = 255;
        int blue = 255;
        Mat tmp = (vec - T);
        Mat trVec = R * (vec - T);

        trVec = C * trVec;

        trVec = trVec / trVec.at<float>(2, 0);

        //		printf("(%d,%d)",(int)trVec.at<float>(0,0),(int)trVec.at<float>(1,0));

        circle(resImg, Point((int)trVec.at<float>(0, 0), (int)trVec.at<float>(1, 0)), 2.0, Scalar(blue, green, red), 2, 8);
    }
}

int main(int argc, const char **argv)
{
    if (argc != 2)
    {
        printf("Usage: FV filename\n");
        exit(0);
    }
    MatrixReaderWriter mrw(argv[1]);
    printf("%d %d\n", mrw.rowNum, mrw.columnNum);

    Mat resImg;

    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.

    int index = 0;

    float v = 1.0;
    float u = 0.5;
    float rad = 100.0;

    resImg = Mat::zeros(600, 800, CV_8UC3);
    drawPoints(mrw, u, v, rad, resImg);
    imshow("Display window", resImg); // Show our image inside it.

    char key;
    while (true)
    {
        key = cvWaitKey(0);
        if (key == 27)
            break;

        switch (key)
        {
        case 'q':
            u += 0.1;
            break;
        case 'a':
            u -= 0.1;
            break;
        case 'w':
            v += 0.1;
            break;
        case 's':
            v -= 0.1;
            break;
        case 'e':
            rad *= 1.1;
            break;
        case 'd':
            rad /= 1.1;
            break;
        }
        resImg = Mat::zeros(600, 800, CV_8UC3);
        drawPoints(mrw, u, v, rad, resImg);
        imshow("Display window", resImg); // Show our image inside it.
    }
}