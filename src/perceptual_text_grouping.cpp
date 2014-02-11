#include "perceptual_text_grouping.h"

/* The text grouping stage takes the regions produced by the text segmentation step
   and makes compact groups of perceptually close or salient regions
*/
void perceptual_text_grouping(Rect A, Rect B){

};

void text_saliency_operator(Rect A, Rect B){
	//saliencey = normal_dist(blob_dimension_ratio()) 
	//            * normal_dist(relative_minimum_distance())
};

// Takes two candidate text regions A and B as inputs
void relative_minimum_distance(Rect A, Rect B){
	// return
	// minimum_distance_btw_regions / (A_minimum_axes + B_maximum_axes)
};

void blob_dimension_ratio(Rect A, Rect B){
	// return 
	// (A_minimum_zxes+A_maximum_axes)/(B_minimum_axes+B_maximum_axes)
};

