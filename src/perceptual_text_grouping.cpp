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

namespace perceptual_text_grouping{

  Mat* perceptual_text_grouping(Mat& image, vector<Rect4f*>& text_region_array){
	// Color to draw  lines
	Scalar delaunay_color(255,255,0);

	// Map of text regions pointers
	std::map<int, Rect4f*> text_region_map;

	// Go throw the text regions and store them in a map
	for(size_t i = 0; i < text_region_array.size(); i++){
		// Get the center point of the region
		Point2f* ptf2f = rect_center_point(text_region_array[i]);
		// x^2+y^2 of the center as unique key for each region
		int xy_sqr = pow(ptf2f->x, 2) + pow(ptf2f->y,2);
		text_region_map.insert(std::make_pair(xy_sqr, text_region_array[i]));
	}

    Size image_size = image.size();
    Rect* image_size_rect = new Rect(0, 0, image_size.width, image_size.height);
    
    Subdiv2D* planarGraph = construct_planar_graph(text_region_array, image_size_rect);

    // Get a vector of edges
    vector<Vec4f> edgeList;
    planarGraph->getEdgeList(edgeList);

    // Draw initial image
    imshow("Threshold Test", image);

    printf("Tap SPACE to iteratively draw lines.\n");

    // For each valid region edge, calculate saliency
    for(size_t i = 0; i < edgeList.size(); i++){
      Vec4f e = edgeList[i];
      try {
    	  // Obtain both regions, if the region is not in the map
    	  // an error is thrown and the points are skipped
    	  Rect4f* A = text_region_map.at(pow(e[0], 2) + pow(e[1],2));
    	  Rect4f* B = text_region_map.at(pow(e[2], 2) + pow(e[3],2));

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
			  printf("x y: %f %f", e[0], e[1]);
			  Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
			  printf("x y: %f %f\n", e[2], e[3]);

			  line(image, pt0, pt1, delaunay_color, 3, 16, 0);
			  imshow("Threshold Test", image);
    	  } else {
    		  vector<Vec4f>::iterator non_salient_edge = edgeList.begin() + i;
    		  edgeList.erase(non_salient_edge);
    	  }
      }
      catch (const std::out_of_range& oor) {
    	  //std::cerr << "Out of Range error: " << oor.what() << '\n';
      }
    }

    // return connected nodes as text regions
    return NULL;
  };

  bool text_saliency_operator(Rect4f* A, Rect4f* B){
    //saliencey = normal_dist(blob_dimension_ratio()) 
    //            * normal_dist(relative_minimum_distance())
    return true;
  };

  // Takes two candidate text regions A and B as inputs
  double relative_minimum_distance(Rect4f* A, Rect4f* B){
    // return
    // minimum_distance_btw_regions / (A_minimum_axes + B_maximum_axes)
    return 0;
  };

  double blob_dimension_ratio(Rect4f* A, Rect4f* B){
    // return 
    // (A_minimum_zxes+A_maximum_axes)/(B_minimum_axes+B_maximum_axes)
    return 0;
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
}

