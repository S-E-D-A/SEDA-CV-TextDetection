#ifndef PERCEPTUAL_TEXT_GROUPING_H
#define PERCEPTUAL_TEXT_GROUPING_H

#include "opencv2/opencv.hpp"

using namespace cv;

/* The text grouping stage takes the regions produced by the text segmentation step
   and makes compact groups of perceptually close or salient regions
*/
void perceptual_text_grouping(Rect A, Rect B);

// Text saliency operator
void text_saliency_operator(Rect A, Rect B);

// Scale-invariant saliency measures
void relative_minimum_distance(Rect A, Rect B);
void blob_dimension_ratio(Rect A, Rect B);

#endif