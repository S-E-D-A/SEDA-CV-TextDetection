#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdlib>
#include <cstdio>
#include <stack>

using namespace cv;

int main(int argc, char ** argv) {

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

  std::stack<int> dfs_contour;
  std::vector<int> text_candidate_contours

  // Find first contour in first hierarchy
  for (int i = 0; i < contours.size(); i++) {
    if (hierarchy[i][1] == -1) {
      dfs_contour.push(i);
      std::cout << "Pushed " << i << std::endl;
      break;
    }
  }

  // Traversal
  while (!dfs_contour.empty()) {

    int curr_contour = dfs_contour.top();
    dfs_contour.pop();

    Mat cont_im = Mat::zeros(src.rows, src.cols, CV_8UC3);
    drawContours(cont_im, contours, curr_contour, Scalar(255,255,255), CV_FILLED, 8);
    imshow("Threshold Test", cont_im);

    if (hierarchy[curr_contour][2] != -1) {
      dfs_contour.push(hierarchy[curr_contour][2]);
    }

    if (hierarchy[curr_contour][0] != -1) {
      dfs_contour.push(hierarchy[curr_contour][0]);
    }

    // Wait for escape keypress
    while (true) {
      int c;
      c = waitKey(20);
      if( (char)c == 27 )
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
