#include "perceptual_text_grouping.h"
#include "gtest/gtest.h"

TEST(Perceptual_text_grouping_test, calculate_region_center_function) {
    Rect_<float>* text_region = new Rect_<float>(5, 5, 21, 21);
    Point2f* calculated_point = perceptual_text_grouping::calculate_region_center(text_region);
    Point2f* center_point = new Point2f(10.5f,10.5f);

    EXPECT_EQ(center_point->x, calculated_point->x);
    EXPECT_EQ(center_point->y, calculated_point->y);

    delete text_region;
    delete calculated_point;
    delete center_point;
}
