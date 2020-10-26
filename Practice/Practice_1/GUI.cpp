#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

#define WIDTH 800
#define HEIGHT 600


Mat image;
int iksz,ipszilon;

void redraw(){
    rectangle(image, Point(0,0),Point(WIDTH,HEIGHT), Scalar(0,0,0),CV_FILLED);
    rectangle(image, Point(iksz,ipszilon),Point(iksz+100,ipszilon+100), Scalar(255,0,0));
    imshow( "Display window", image );                   // Show our image inside it.
}

void MouseCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        iksz=x;
        ipszilon=y;
        redraw();
    }

}

int main( int argc, char** argv )
{
    image=Mat::zeros(600,800,CV_8UC3);
    iksz=-100.0;
    ipszilon=-100.0;



    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    setMouseCallback("Display window", MouseCallBackFunc, NULL);



    imshow( "Display window", image );                   // Show our image inside it.


  int key;
  while(true){
    key=cvWaitKey(100);
    iksz++;
    if(key==27) break;

    switch(key){
      case 'o':
            iksz--;
            break;
      case 'p':
            iksz++;
        break;
      case 'q':
            ipszilon--;
        break;
      case 'a':
            ipszilon++;
        break;
    }
    redraw();
  }


    return 0;
}