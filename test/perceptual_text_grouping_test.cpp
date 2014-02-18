#include "perceptual_text_grouping.h"
#include "gtest/gtest.h"

typedef Rect_<float> Rect4f;

TEST(Perceptual_text_grouping_test, rect_center_point_function) {
	float x = 7.f, y = 3.f, width = 19.f, height = 5.f;
    Rect_<float>* text_region = new Rect4f(x, y, width, height);
    Point2f* calculated_point = perceptual_text_grouping::rect_center_point(text_region);
    Point2f* center_point = new Point2f(x+width/2, y+height/2);

    EXPECT_EQ(center_point->x, calculated_point->x);
    EXPECT_EQ(center_point->y, calculated_point->y);

    delete text_region;
    delete calculated_point;
    delete center_point;
}
