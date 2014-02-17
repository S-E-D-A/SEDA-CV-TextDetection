#include "perceptual_text_grouping.h"

namespace perceptual_text_grouping{

  Mat* perceptual_text_grouping(Mat* image, vector<Rect4f*> text_region_array){
    // construct_planar_graph
    // for edge e in graph
    //   if !text_saliency_operator remove edge
    // return connected nodes as text regions
    return NULL;
  };

  bool text_saliency_operator(Rect4f* A, Rect4f* B){
    //saliencey = normal_dist(blob_dimension_ratio()) 
    //            * normal_dist(relative_minimum_distance())
    return false;
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

  Subdiv2D* construct_planar_graph(vector<Rect4f*> text_region_array, 
    Rect image_size){

    // Initialize graph with image size the maximum bounding box
    Subdiv2D* planar_graph = new Subdiv2D(image_size);

    MemStorage storage(cvCreateMemStorage(0));

    // Insert each text region center into the graph
    for(unsigned int i = 0; i < text_region_array.size(); i++){
     Point2f* tmp_point = rect_center_point(text_region_array[i]);
     planar_graph->insert(*tmp_point);
    }

    return planar_graph;
  };

  Point2f* rect_center_point(Rect4f* text_region){
    // returns the center of a Rect4f as a Point2f
    return new Point2f(text_region->width/2, text_region->height/2);
  };
}

