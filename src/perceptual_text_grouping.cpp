#include "perceptual_text_grouping.h"

namespace perceptual_text_grouping{
	/* The text grouping stage takes the regions produced by the text segmentation step
	   and makes compact groups of perceptually close or salient regions
	*/
	Mat* perceptual_text_grouping(Mat* image, Rect* text_region_array){
		// construct_planar_graph
		// for edge e in graph
		//   if !text_saliency_operator remove edge
		// return connected nodes as text regions
		return NULL;
	};

	bool text_saliency_operator(Rect A, Rect B){
		//saliencey = normal_dist(blob_dimension_ratio()) 
		//            * normal_dist(relative_minimum_distance())
		return false;
	};

	// Takes two candidate text regions A and B as inputs
	double relative_minimum_distance(Rect A, Rect B){
		// return
		// minimum_distance_btw_regions / (A_minimum_axes + B_maximum_axes)
		return 0;
	};

	double blob_dimension_ratio(Rect A, Rect B){
		// return 
		// (A_minimum_zxes+A_maximum_axes)/(B_minimum_axes+B_maximum_axes)
		return 0;
	};

	CvSubdiv2D* construct_planar_graph(Rect* text_region_array){
		// for i in text_region_array
		// 	add to CvSubdiv2D Delaunay graph(calculate_region_center(i))
		return NULL;
	};

	Point2f* calculate_region_center(Rect* text_region){
		Point2f* ret = new Point2f(text_region->width/2, text_region->height/2);
		return ret;
	}
}

