#ifndef TEXT_CANDIDATE_DETECTION_HPP
#define TEXT_CANDIDATE_DETECTION_HPP

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <set>

using namespace cv;

namespace text_candidate_detection {

  std::vector<Rect *> text_candidate_detection(Mat src);

  double border_energy(vector<Point> & points, Mat & sobel_image);
  double eigen_texture_measure(vector<Point> & points, Mat im, int n, int l); 
  bool size_filter(Rect contour_bb, const Mat& src);
}

#endif
