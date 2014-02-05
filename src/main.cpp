#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdlib>
#include <cstdio>
#include <stack>

using namespace cv;

int main(int argc, char ** argv) {
  /* Handles input otpions */
  /* ##################### */
  extern char *optarg;
  int c, err = 0; 

  // Whether or not -i flag set to value
  int image_file=0;
  // Store the -i flag input value
  char *image_file_name=NULL;

  // Usage string of this program
  static char usage[] = "Usage: %s -i path/to/image\n";

  while ((c = getopt(argc, argv, "hi:")) != -1)
    switch (c) {
    case 'i':
      image_file = 1;
      // Store the value of -i flag in variable
      image_file_name = optarg;
      break;
    case 'h':
      // Show help info and exit
      fprintf(stderr, usage, argv[0]);
      exit(1);
    case '?':
      // Given incorrect option flag error
      err = 1;
      break;
    }
  
  if (image_file == 0) { /* -i  mandatory */
    fprintf(stderr, "%s: missing -i option\n", argv[0]);
    fprintf(stderr, usage, argv[0]);
    exit(1);
  } else if (err) {
    fprintf(stderr, usage, argv[0]);
    exit(1);
  }
  /* ##################### */

  Mat src = imread(image_file_name, 1);
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
  std::vector<int> text_candidate_contours;

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

    if (hierarchy[curr_contour][0] != -1) {
      dfs_contour.push(hierarchy[curr_contour][0]);
    }

    if (hierarchy[curr_contour][2] != -1) {
      dfs_contour.push(hierarchy[curr_contour][2]);
    }

    // Wait for escape keypress
    while (true) {
      int c;
      c = waitKey(20);
      if( (char)c == 27 )
        { break; }
    } 

  }

}
