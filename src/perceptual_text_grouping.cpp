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
#include <set>
#include <algorithm>

// Whether or not to build delaunay triangulation line by line
const bool LINE_BY_LINE_MODE = false;

const float N_CONST = 1/sqrt(2*M_PI);
const float SALIENT_THRESHOLD = 0.01f;

const float RELATIVE_MIN_AVG = 1;
const float RELATIVE_MIN_SD = 2;
const float BLOB_DIM_AVG = 0;
const float BLOB_DIM_SD = 4;

typedef std::map<int, Rect*> RectMap;
typedef std::set<Rect*> RectPtrSet;

namespace perceptual_text_grouping{

  vector<Rect*> perceptual_text_grouping(Mat& image, vector<Rect*>& text_region_array){
	// Color to draw  lines
 	Scalar delaunay_color(255,255,0);
	Scalar bad_color(0,0,255);

	// Map of text regions pointers
	RectMap text_region_map;

	// Go through the text regions and store them in a map
	for(size_t i = 0; i < text_region_array.size(); i++){
		// Get the center point of the region
		Point2f* center_pt = rect_center_point(text_region_array[i]);
		// x^2+y^2 of the center as unique key for each region
		int xy_sqr = pow(cvRound(center_pt->x), 2) + pow(cvRound(center_pt->y),2);
		text_region_map.insert(std::make_pair(xy_sqr, text_region_array[i]));
	}

	// Get the size of the image and create a Rect from it
    Size image_size = image.size();
    Rect image_size_rect(0, 0, image_size.width, image_size.height);
    
    // Construct a planar graph and maintain a pointer to it
    Subdiv2D* planarGraph = construct_planar_graph(text_region_array, &image_size_rect);

    // Get a vector of edges
    vector<Vec4f> edgeList;
    planarGraph->getEdgeList(edgeList);

    vector<Vec4f> realEdgeList;

    // Remove non valid regions created during delaunay construction
    // For each valid region edge, calculate saliency
	for(size_t i = 0; i < edgeList.size(); i++){
	  Vec4f e = edgeList[i];
	  try {
		  // Obtain both regions, if the region is not in the map
		  // an error is thrown and the points are skipped
		  int xy_sqr_1 = pow(cvRound(e[0]),2) + pow(cvRound(e[1]),2);
		  text_region_map.at(xy_sqr_1);

		  int xy_sqr_2 = pow(cvRound(e[2]),2) + pow(cvRound(e[3]),2);
		  text_region_map.at(xy_sqr_2);

		  // Real edge
		  realEdgeList.push_back(e);
	  } catch (const std::out_of_range& oor) {
		  // This is thrown and ignored if an out of bounds
		  // rect from the Delaunay calculations is used
	  }
	}

    // Draw initial image
    imshow("Threshold Test", image);

    if(LINE_BY_LINE_MODE){
        printf("Tap SPACE to iteratively draw lines.\n\n");
    }

    vector<Vec4f> salientEdgeList;

    // For each valid region edge, calculate saliency
    for(size_t i = 0; i < realEdgeList.size(); i++){
      Vec4f e = realEdgeList[i];

      // Obtain both regions, if the region is not in the map
	  // an error is thrown and the points are skipped
	  int xy_sqr_1 = pow(cvRound(e[0]),2) + pow(cvRound(e[1]),2);
	  Rect* A = text_region_map.at(xy_sqr_1);

	  int xy_sqr_2 = pow(cvRound(e[2]),2) + pow(cvRound(e[3]),2);
	  Rect* B = text_region_map.at(xy_sqr_2);

	  if(text_saliency_operator(A,B)){
		  if(LINE_BY_LINE_MODE){
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
		  }

		  Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
		  Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));

		  salientEdgeList.push_back(e);

		  line(image, pt0, pt1, delaunay_color, 1, 16, 0);
		  rectangle(image, *A, delaunay_color, 1, 16, 0);
		  rectangle(image, *B, delaunay_color, 1, 16, 0);

		  imshow("Threshold Test", image);
	  }
    }

    // The vector of all sets of connected rects
    std::vector<RectPtrSet*> edge_groups;

    // Store all connected Rects as groups inside of sets
    for(size_t i = 0; i < salientEdgeList.size(); i++){
    	Vec4f e = salientEdgeList[i];

    	int xy_sqr_1 = pow(cvRound(e[0]),2) + pow(cvRound(e[1]),2);
    	Rect* A = text_region_map.at(xy_sqr_1);

    	int xy_sqr_2 = pow(cvRound(e[2]),2) + pow(cvRound(e[3]),2);
    	Rect* B = text_region_map.at(xy_sqr_2);

    	bool found = false;

    	// Counter of groups which contain the same rects
    	vector<int> region_counter;

    	for(size_t j = 0; j < edge_groups.size(); j++){
    		if(std::find(edge_groups[j]->begin(), edge_groups[j]->end(), A) != edge_groups[j]->end() ||
    		   std::find(edge_groups[j]->begin(), edge_groups[j]->end(), B) != edge_groups[j]->end()) {
    			found = true;
    			edge_groups[j]->insert(A);
    			edge_groups[j]->insert(B);
    			region_counter.push_back(j);
    		}
    	}

    	if(!found){
    		std::set<Rect*>* insert_set = new std::set<Rect*>();
    		insert_set->insert(A);
    		insert_set->insert(B);
    		edge_groups.push_back(insert_set);
    	} else if(region_counter.size() > 1){
    		// If there were matches in more than one group
    		// it means two existing groups are connected and need to be merged
    		RectPtrSet* merged_set;
    		std::vector<RectPtrSet*> sets_to_delete;

    		//
    		for(size_t i = 0; i < region_counter.size(); i++){
    			int region_index = region_counter[i];

    			sets_to_delete.push_back(edge_groups[region_index]);

    			RectPtrSet* edge_set_ptr = edge_groups[region_index];
    			RectPtrSet mergeInSet = *edge_set_ptr;

    			for(RectPtrSet::iterator it = mergeInSet.begin(); it!=mergeInSet.end(); ++it){
    				merged_set->insert(*it);
    			}
    		}

    		for(size_t i = 0; i < sets_to_delete.size(); i++){
    			std::vector<RectPtrSet*>::iterator delete_it = std::find(edge_groups.begin(), edge_groups.end(), sets_to_delete[i]);
    			edge_groups.erase(delete_it);
    		}
    		edge_groups.push_back(merged_set);
    	}
    }

    vector<Rect*> final_text_region_rects;

	// Get the enclosing rects for the text regions
    for(size_t i = 0; i < edge_groups.size(); i++){
    	vector<int> widths;
    	vector<int> heights;

    	for(std::set<Rect*>::iterator it=edge_groups[i]->begin(); it!=edge_groups[i]->end(); ++it){
    		widths.push_back((*it)->x);
    		widths.push_back((*it)->x+(*it)->width);
    		heights.push_back((*it)->y);
    		heights.push_back((*it)->y+(*it)->height);
    	}

    	std::sort(widths.begin(), widths.end());
    	std::sort(heights.begin(), heights.end());

    	int min_x = *(widths.begin());
    	int min_y = *(heights.begin());

    	int width = *(widths.end() -1 ) - min_x;
    	int height = *(heights.end() -1) - min_y;

    	Rect* max_container_rect = new Rect(min_x, min_y, width, height);
    	final_text_region_rects.push_back(max_container_rect);

    	rectangle(image, *max_container_rect, bad_color, 1, 16, 0);
    }

    for(size_t j = 0; j < edge_groups.size(); j++){
    	delete edge_groups[j];
    	edge_groups[j] = NULL;
    }

    delete planarGraph;
    planarGraph = NULL;

    return final_text_region_rects;
  };

  bool text_saliency_operator(Rect* A, Rect* B){
	float rmd = relative_minimum_distance(A,B);
	float bdr = blob_dimension_ratio(A,B);

	float rmd_n = normal_distribution(rmd,
			RELATIVE_MIN_AVG,RELATIVE_MIN_SD);

	float bdr_n = normal_distribution(bdr,
			BLOB_DIM_AVG,BLOB_DIM_SD);

 	return (rmd_n*bdr_n) > SALIENT_THRESHOLD;
  };

  // Takes two candidate text regions A and B as inputs
  float relative_minimum_distance(Rect* A, Rect* B){
    return min_distance(A,B)/(axis(A,minAxis) + axis(B,minAxis));
  };

  float blob_dimension_ratio(Rect* A, Rect* B){
	return (axis(A,minAxis)+axis(A,maxAxis))/(axis(B,minAxis)+axis(B,maxAxis));
  };

  Subdiv2D* construct_planar_graph(vector<Rect*> &text_region_array,
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

	Point2f* rect_center_point(Rect* text_region){
	  // returns the center of a Rect as a Point
	  return new Point2f(cvRound((text_region->x + text_region->width/2)),
			  cvRound((text_region->y + text_region->height/2)));
	};

  float normal_distribution(float x, float mu, float sigma){
	  return N_CONST*(1/sigma) * exp(-powf((x-mu),2)/(2*powf(sigma,2)));
  };

  float axis(Rect* A, bool (*which_axis)(Rect*)){
 	  float height_axis = A->height;
 	  float width_axis = A->width;

 	  if(which_axis(A)){
 		  return height_axis;
 	  } else {
 		  return width_axis;
 	  }
   };

  bool minAxis(Rect* A){
	  return A->height < A->width;
  };

  bool maxAxis(Rect* A){
  	  return A->height > A->width;
  };

  float min_distance(Rect* A, Rect* B){

	  Point2f* A_pt = rect_center_point(A);
	  Point2f* B_pt = rect_center_point(B);

	  Rect intersect = *A&*B;

	  // If the rectangles overlap they have 0 distance between them
	  if(intersect.width > 0 ||
	     intersect.height > 0 ||
		 intersect.x > 0 ||
		 intersect.y > 0) {
		  return 0;
	  }

	  return sqrt(pow((cvRound(A_pt->x) - cvRound(B_pt->x)),2) +
			  pow((cvRound(A_pt->y) - cvRound(B_pt->y)),2));
  };
}

