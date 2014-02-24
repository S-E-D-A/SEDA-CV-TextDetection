#ifndef PERCEPTUAL_TEXT_GROUPING_H
#define PERCEPTUAL_TEXT_GROUPING_H

#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;

namespace perceptual_text_grouping {
  /* The text grouping stage takes the regions produced by the text
     segmentation step and makes compact groups of perceptually 
     close or salient regions
  */
  vector<Rect*> perceptual_text_grouping(Mat& image, vector<Rect*>& text_region_array);

  // Text saliency operator
  bool text_saliency_operator(Rect* A, Rect* B);

  // Scale-invariant saliency measures
  float relative_minimum_distance(Rect* A, Rect* B);
  float blob_dimension_ratio(Rect* A, Rect* B);

  // Creation of planar graph using Delaunay triangulation
  Subdiv2D* construct_planar_graph(vector<Rect*> &text_region_array,
                            Rect* image_size);
  Point2f* rect_center_point(Rect* text_region);

  float normal_distribution(float x, float mu, float sigma);

  // Expects minAxis or maxAxis as inputs to which_axis
  float axis(Rect* A, bool (*which_axis)(Rect*));
  bool minAxis(Rect* A);
  bool maxAxis(Rect* A);

  float min_distance(Rect* A, Rect* B);
}

#endif
