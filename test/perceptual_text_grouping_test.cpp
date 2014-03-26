#include "perceptual_text_grouping.hpp"
#include "gtest/gtest.h"

TEST(Perceptual_text_grouping_test, rect_center_point_function) {
	int x = 7, y = 3, width = 19, height = 5;
    Rect* text_region = new Rect(x, y, width, height);
    Point2f* calculated_point = perceptual_text_grouping::rect_center_point(text_region);
    Point2f* center_point = new Point2f(x+width/2, y+height/2);

    EXPECT_EQ(center_point->x, calculated_point->x);
    EXPECT_EQ(center_point->y, calculated_point->y);

    delete text_region;
    delete calculated_point;
    delete center_point;
}

TEST(Perceptual_text_grouping_test, construct_planar_graph_function) {
  vector<Rect*> text_region_array;
  Rect* image_size = new Rect(0,0,600,600);
  
  perceptual_text_grouping::construct_planar_graph(text_region_array, 
    image_size);

  delete image_size;
}

TEST(Perceptual_text_grouping_test, perceptual_text_grouping_random_regions){
	Mat image = imread("images/test.png", 1);

	vector<Rect*> text_region_array;

	// Get the size of the image and create a Rect from it
	Size image_size = image.size();

	int num_ints = rand() % 15;

	for(int i = 0; i < num_ints; i++){
	  int x = rand() % image_size.width;
	  int y = rand() % image_size.height;

	  int remainder_width = image_size.width - x ;
	  int remainder_height = image_size.height - y;

	  int bounding_box_x = rand()%remainder_width;
	  int bounding_box_y = rand()%remainder_height;

	  Rect* r1 = new Rect(x, y, bounding_box_x, bounding_box_y);
	  text_region_array.push_back(r1);
	}

	perceptual_text_grouping::perceptual_text_grouping(image, text_region_array);

	imshow("Test Window", image);

	printf("Press ESC or close window to exit.\n");

	// Wait for escape keypress
	while (true) {
	  int c;
	  c = waitKey(20);
	  if( (char)c == 27 )
		{ break; }

    // If user closes window
    if (!cvGetWindowHandle("Test Window"))
      break;
	}
}
TEST(Perceptual_text_grouping_test, perceptual_text_grouping_function) {
	Mat image = imread("images/test.png", 1);

	vector<Rect*> text_region_array;

	Rect* r1 = new Rect(202, 52, 44, 70);
	text_region_array.push_back(r1);
	r1 = new Rect(280, 52, 44, 70);
	text_region_array.push_back(r1);
	r1 = new Rect(354, 52, 44, 70);
	text_region_array.push_back(r1);

	r1 = new Rect(244, 391, 79, 97);
	text_region_array.push_back(r1);
	r1 = new Rect(384, 391, 16, 100);
	text_region_array.push_back(r1);
	r1 = new Rect(455, 391, 58, 102);
	text_region_array.push_back(r1);

	r1 = new Rect(247, 226, 54, 68);
	text_region_array.push_back(r1);
	r1 = new Rect(331, 226, 48, 68);
	text_region_array.push_back(r1);
	r1 = new Rect(411, 226, 63, 68);
	text_region_array.push_back(r1);

	perceptual_text_grouping::perceptual_text_grouping(image, text_region_array);

	imshow("Test Window", image);

	printf("Press ESC or close window to exit.\n");

	// Wait for escape keypress
	while (true) {
	  int c;
	  c = waitKey(20);
	  if( (char)c == 27 )
	    { break; }

    // If user closes window
    if (!cvGetWindowHandle("Test Window"))
      break;
	}
}
