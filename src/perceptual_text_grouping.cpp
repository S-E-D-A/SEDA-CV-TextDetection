#include "../include/perceptual_text_grouping.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <stddef.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

const float N_CONST = 1/sqrt(2*M_PI);
const float SALIENT_THRESHOLD = 0.01f;

const float RELATIVE_MIN_AVG = 1;
const float RELATIVE_MIN_SD = 2;
const float BLOB_DIM_AVG = 0;
const float BLOB_DIM_SD = 4;

namespace perceptual_text_grouping{

  void perceptual_text_grouping(Mat& image, vector<Rect4f*>& text_region_array){
	// Color to draw  lines
	Scalar delaunay_color(255,255,0);
	Scalar bad_color(0,0,255);

	// Map of text regions pointers
	std::map<int, Rect4f*> text_region_map;

	// Go through the text regions and store them in a map
	for(size_t i = 0; i < text_region_array.size(); i++){
		// Get the center point of the region
		Point2f* center_pt = rect_center_point(text_region_array[i]);
		// x^2+y^2 of the center as unique key for each region
		int xy_sqr = pow(center_pt->x, 2) + pow(center_pt->y,2);
		printf("xy_sqr %i\n", xy_sqr);
		text_region_map.insert(std::make_pair(xy_sqr, text_region_array[i]));
	}

	// Get the size of the image and create a Rect from it
    Size image_size = image.size();
    Rect* image_size_rect = new Rect(0, 0, image_size.width, image_size.height);
    
    // Construct a planar graph and maintain a pointer to it
    Subdiv2D* planarGraph = construct_planar_graph(text_region_array, image_size_rect);

    // Get a vector of edges
    vector<Vec4f> edgeList;
    planarGraph->getEdgeList(edgeList);

    // Draw initial image
    imshow("Threshold Test", image);

    printf("Tap SPACE to iteratively draw lines.\n\n");

    // For each valid region edge, calculate saliency
    for(size_t i = 0; i < edgeList.size(); i++){
      Vec4f e = edgeList[i];
      try {
    	  // Obtain both regions, if the region is not in the map
    	  // an error is thrown and the points are skipped
    	  int xy_sqr_1 = pow(e[0],2) + pow(e[1],2);
    	  printf("xy_sqr_1 %i\n", xy_sqr_1);
    	  Rect4f* A = text_region_map.at(xy_sqr_1);
    	  int xy_sqr_2 = pow(e[2],2) + pow(e[3],2);
    	  printf("xy_sqr_2 %i\n", xy_sqr_2);
    	  Rect4f* B = text_region_map.at(xy_sqr_2);

    	  if(text_saliency_operator(A,B)){
			  // Wait for keypress
			  while (true) {
				   int c = waitKey(120);
				   if(c == 32 ){ //Space
					   // Drawline
					   break;
				   } else if (c == 27){ //Escape
					   // End program
					   exit(1);
				   }
			  }

			  Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
			  Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));

			  line(image, pt0, pt1, delaunay_color, 1, 16, 0);
			  rectangle(image, *A, delaunay_color, 1, 16, 0);
			  rectangle(image, *B, delaunay_color, 1, 16, 0);

			  imshow("Threshold Test", image);
    	  }
      } catch (const std::out_of_range& oor) {
    	  // This is thrown and ignored if an out of bounds
      	  // rect from the Delaunay calculations is used
      }
    }

    // Delete current region pointers
    for(size_t i = 0; i < text_region_array.size(); i++){
    	delete text_region_array[i];
    }
    // traverse planar graph for all > 1 connected rects
    //   find max and min x,y of connected Rects
    //   store the containing Rect region in textRegions vector
  };

  bool text_saliency_operator(Rect4f* A, Rect4f* B){
	float rmd = relative_minimum_distance(A,B);
	float bdr = blob_dimension_ratio(A,B);

	float rmd_n = normal_distribution(rmd,
			RELATIVE_MIN_AVG,RELATIVE_MIN_SD);

	float bdr_n = normal_distribution(bdr,
			BLOB_DIM_AVG,BLOB_DIM_SD);


	float f1 = rmd_n*bdr_n;
	float f2 = SALIENT_THRESHOLD;

	bool ret = f1 > f2;

	if(ret){
		printf("\nSALIENT REGION\n");
		printf("Relative minimum distance: %f\n", rmd);
		printf("Blob dimension ratio: %f\n", bdr);
		printf("Saliency score: %e\n", f1);
	} else {
		printf("\nNONSALIENT REGION\n");
		printf("Relative minimum distance: %f\n", rmd);
		printf("Blob dimension ratio: %f\n", bdr);
		printf("Saliency score: %e\n\n", f1);
	}
 	return ret;
  };

  // Takes two candidate text regions A and B as inputs
  float relative_minimum_distance(Rect4f* A, Rect4f* B){
    return min_distance(A,B)/(axis(A,minAxis) + axis(B,minAxis));
  };

  float blob_dimension_ratio(Rect4f* A, Rect4f* B){
	return (axis(A,minAxis)+axis(A,maxAxis))/(axis(B,minAxis)+axis(B,maxAxis));
  };

  Subdiv2D* construct_planar_graph(vector<Rect4f*> &text_region_array,
	Rect *image_size){

	// Initialize graph with image size the maximum bounding box
	Subdiv2D* planar_graph = new Subdiv2D(*image_size);

	// Insert each text region center into the graph
	for(unsigned int i = 0; i < text_region_array.size(); i++){
		Point2f* center_point = rect_center_point(text_region_array[i]);
		planar_graph->insert(*center_point);
	}

	return planar_graph;
  };

	Point2f* rect_center_point(Rect4f* text_region){
	  // returns the center of a Rect4f as a Point2f
	  return new Point2f(text_region->x + text_region->width/2,
					   text_region->y + text_region->height/2);
	};

  float normal_distribution(float x, float mu, float sigma){
	  return N_CONST*(1/sigma) * exp(-powf((x-mu),2)/(2*powf(sigma,2)));
  };

  float axis(Rect4f* A, bool (*which_axis)(Rect4f*)){
 	  float height_axis = A->height;
 	  float width_axis = A->width;

 	  if(which_axis(A)){
 		  return height_axis;
 	  } else {
 		  return width_axis;
 	  }
   };

  bool minAxis(Rect4f* A){
	  return A->height < A->width;
  };

  bool maxAxis(Rect4f* A){
  	  return A->height > A->width;
  };

  float min_distance(Rect4f* A, Rect4f* B){

	  Point2f* A_pt = rect_center_point(A);
	  Point2f* B_pt = rect_center_point(B);

	  Rect4f intersect = *A&*B;

	  // If the rectangles overlap they have 0 distance between them
	  if(intersect.width > 0 ||
	     intersect.height > 0 ||
		 intersect.x > 0 ||
		 intersect.y > 0) {
		  return 0;
	  }

	  printf("Min dist: %f\n", sqrtf(powf((A_pt->x - B_pt->x),2) + powf((A_pt->y - B_pt->y),2)));
	  return sqrtf(powf((A_pt->x - B_pt->x),2) + powf((A_pt->y - B_pt->y),2));
  };
}

