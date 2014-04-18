#ifndef __RECOGNIZE_TEXT_H
#define __RECOGNIZE_TEXT_H

#include <vector>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "er_filter.hpp"


using namespace cv;
using namespace std;

namespace recognize_text
{
	void recognize_text(Mat &src);
	void components_draw(Mat &src, vector<ERStat> &comps);
	void groups_draw(Mat &src, vector<Rect> &groups);
	void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions);
}

#endif
