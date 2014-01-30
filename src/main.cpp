#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdlib>
#include <cstdio>

using namespace cv;

int main(int argc, char ** argv) {

  if (argc < 2) {
    std::cout << "Please add an image file" << std::endl;
    return 1;
  }

  Mat src = imread(argv[1], 1);
  Mat src_gray;
  cvtColor(src, src_gray, CV_RGB2GRAY);

  namedWindow("Threshold Test", CV_WINDOW_AUTOSIZE);

  Mat thresh_im;
  // threshold(src_gray,thresh_im, 127, 255, THRESH_BINARY);
  adaptiveThreshold(src_gray, thresh_im, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 17, 0);

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(thresh_im, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

  // breadth first traversal
  //for (auto h = hierarchy.begin(); h != hierarchy.end(); h++) {
  for (size_t h = 0; h < hierarchy.size(); ++h) {
    std::cout << h << std::endl;

      Mat cc = Mat::zeros(src.size(), src.type());
      drawContours(cc, contours, h, 255, CV_FILLED, 8, hierarchy);
      imshow("Threshold Test", cc); 
      

      // Pause
      while(true)
      {
        int c;
        c = waitKey( 32 );
        if( (char)c == 32 ) // Type the space bar to advance
          { break; }
      }

  }

  imshow("Threshold Test", thresh_im);

  while(true)
  {
    int c;
    c = waitKey( 20 );
    if( (char)c == 27 )
      { break; }
   }
  return 0;
}
