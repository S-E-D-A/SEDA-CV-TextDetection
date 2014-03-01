#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <stack>
#include <set>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "perceptual_text_grouping.hpp"
#include "text_candidate_detection.hpp"

using namespace cv;

double border_energy(vector<Point> & points, Mat & sobel_image);
double eigen_texture_measure(vector<Point> & points, Mat im, int n, int l); 

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

  std::vector<Rect *> text_bounding_boxes;
  text_bounding_boxes = text_candidate_detection::text_candidate_detection(src);

/*
  Mat cont_im = Mat::zeros(src.rows, src.cols, CV_8UC3);
  for (auto it = text_candidate_contours.begin(); it != text_candidate_contours.end(); ++it) {
    // Show it
    drawContours(cont_im, contours, *it, Scalar(255,255,255), CV_FILLED, 8);
  }
  cont_im = ~cont_im;
*/
  namedWindow("Threshold Test", CV_WINDOW_AUTOSIZE);
  
  perceptual_text_grouping::perceptual_text_grouping(src, text_bounding_boxes);
  
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

  //std::cout << "BE: " << border_energy << std::endl;

  return border_energy;
}

// Computes the Eigen-Transform texture operator for the image block im
// l : number of eigenvalues to disgard
double eigen_texture_measure(vector<Point> & points, Mat im, int n, int l) {

  Rect contour_bb = boundingRect(points);
  int w = round(contour_bb.size().height / 4.0);

  double texture_measure = 0;
  for (int i = 0; i < n; ++i) {

    int idx = rand() % points.size(); 
    std::cout << idx << std::endl;

    Point sample = points[idx];

    if ((sample.x - w < 0) || (sample.x + w >= im.size().width)) {
      --n;
      continue;
    }

    if ((sample.y - w < 0) || (sample.y + w >= im.size().height)) {
      --n;
      continue;
    }
     
    Rect roi(sample.x - w, sample.y - w, w, w); 

    Mat image_roi = im(roi);
    Mat image_roi_float;
    image_roi.convertTo(image_roi_float, CV_32F);

    std::vector<double> singular_vals; 

    // We only need the singular values
    SVD::compute(image_roi_float, singular_vals, SVD::NO_UV); 

    int w = singular_vals.size();

    double eigen_val_sum = 0;
    for (int j = l; j < w ; ++j) {
      eigen_val_sum += singular_vals[j];
    }
    texture_measure += eigen_val_sum / (1 + w - l);
  }

  std::cout << "ET: " << texture_measure << std::endl;
  if (n == 0)
    return 0;
  texture_measure /= n;
  return texture_measure;
}
