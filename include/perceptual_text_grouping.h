#ifndef PERCEPTUAL_TEXT_GROUPING_H
#define PERCEPTUAL_TEXT_GROUPING_H

#include "opencv2/opencv.hpp"

using namespace cv;

/* The text grouping stage takes the regions produced by the text segmentation step
   and makes compact groups of perceptually close or salient regions
*/
Mat* perceptual_text_grouping(Rect* text_region_array);

// Text saliency operator
bool text_saliency_operator(Rect A, Rect B);

// Scale-invariant saliency measures
double relative_minimum_distance(Rect A, Rect B);
double blob_dimension_ratio(Rect A, Rect B);

// Creation of planar graph using Delaunay triangulation
CvSubdiv2D* construct_planar_graph(Rect* text_region_array);
Point2f* calculate_region_center(Rect region);

#endif