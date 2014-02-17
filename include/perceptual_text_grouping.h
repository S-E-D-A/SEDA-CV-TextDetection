#ifndef PERCEPTUAL_TEXT_GROUPING_H
#define PERCEPTUAL_TEXT_GROUPING_H

#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;

typedef Rect_<float> Rect4f;

namespace perceptual_text_grouping {
  /* The text grouping stage takes the regions produced by the text 
     segmentation step and makes compact groups of perceptually 
     close or salient regions
  */
  Mat* perceptual_text_grouping(Mat* image, vector<Rect4f*> text_region_array);

  // Text saliency operator
  bool text_saliency_operator(Rect4f* A, Rect4f* B);

  // Scale-invariant saliency measures
  double relative_minimum_distance(Rect4f* A, Rect4f* B);
  double blob_dimension_ratio(Rect4f* A, Rect4f* B);

  // Creation of planar graph using Delaunay triangulation
  Subdiv2D* construct_planar_graph(vector<Rect4f*> text_region_array, 
                            Rect image_size);
  Point2f* rect_center_point(Rect4f* text_region);
}

#endif