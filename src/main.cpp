#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdlib>
#include <cstdio>
#include <stack>
#include <set>

using namespace cv;

double border_energy(vector<Point> & points, Mat & sobel_image);
double eigen_texture_measure(Mat im, int l); 

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

  // Generate Sobel Gradient Image
  Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
  Sobel(src_gray, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  Sobel(src_gray, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);

  // Grad 
  Mat sobel_magnitude;
  Mat sobel_angle;
  cartToPolar(grad_x,grad_y,sobel_magnitude,sobel_angle);

  std::stack<int> dfs_contour;
  std::set<int> text_candidate_contours;

  // Find first contour in first hierarchy
  for (size_t i = 0; i < contours.size(); i++) {
    if (hierarchy[i][1] == -1) {
      dfs_contour.push(i);
      std::cout << "Pushed " << i << std::endl;
      break;
    }
  }

  // Traversal
  while (!dfs_contour.empty()) {

    // Pop current contour
    int curr_contour = dfs_contour.top();
    dfs_contour.pop();


    /*
    *  Region filtering
    */ 

    // Size
    Rect contour_bb = boundingRect(contours[curr_contour]);
    std::cout <<  contour_bb.size().area() << std::endl;
    if (contour_bb.size().area() > 0.5*src.rows*src.cols) {
//      std::cout << "rejecting region: too large" << std::endl;
    } else if (contour_bb.size().area() < 30) {
//      std::cout << "rejecting region: too small" << std::endl;
    } else if (border_energy(contours[curr_contour], sobel_magnitude) < 30.0) {
      std::cout << "rejecting region: border energy too low" << std::endl;
    } else {

    // Show it
/*
    Mat cont_im = Mat::zeros(src.rows, src.cols, CV_8UC3);
    drawContours(cont_im, contours, curr_contour, Scalar(255,255,255), CV_FILLED, 8);
    imshow("Threshold Test", cont_im);
    while (true) {
      int c;
      c = waitKey(20);
      if( (char)c == 27 )
        { break; }
    } 
*/
      // Add to candidates
      text_candidate_contours.insert(curr_contour);

      // Remove parents
      int parent_contour = hierarchy[curr_contour][3];
      text_candidate_contours.erase(parent_contour);
/*
      while (parent_contour != -1) {
        text_candidate_contours.erase(parent_contour);
        parent_contour = hierarchy[parent_contour][3];
      }
*/
    }

    // Push neighbor
    if (hierarchy[curr_contour][0] != -1) {
      dfs_contour.push(hierarchy[curr_contour][0]);
    }

    // Push Children (This gets traversed first, if possible)
    if (hierarchy[curr_contour][2] != -1) {
      dfs_contour.push(hierarchy[curr_contour][2]);
    }

  }

  Mat cont_im = Mat::zeros(src.rows, src.cols, CV_8UC3);
  for (auto it = text_candidate_contours.begin(); it != text_candidate_contours.end(); ++it) {
    // Show it
    drawContours(cont_im, contours, *it, Scalar(255,255,255), CV_FILLED, 8);
  }
  cont_im = ~cont_im;
  imshow("Threshold Test", cont_im);
  // Wait for escape keypress
  while (true) {
    int c;
    c = waitKey(20);
    if( (char)c == 27 )
      { break; }
  } 
}


double border_energy(vector<Point> & points, Mat & sobel_image) {

  double border_energy = 0.0;
  for (auto itr = points.begin(); itr != points.end(); ++itr) {
    border_energy += (float)sobel_image.at<float>(*itr);
  }

  border_energy = border_energy / (double)points.size();

  std::cout << "BE: " << border_energy << std::endl;

  return border_energy;
}

// Computes the Eigen-Transform texture operator for the image block im
// l : number of eigenvalues to disgard
double eigen_texture_measure(Mat im, int l) {
    // We only need the singular values
    std::vector<double> singular_vals; 
    SVD::compute(im, singular_vals, SVD::NO_UV); 

    int w = singular_vals.size();

    double texture_measure = 0;
    for (; l < w ; ++l) {
      texture_measure += singular_vals[l];
    }
    texture_measure /= 1 + w - l;
    return texture_measure;
}
