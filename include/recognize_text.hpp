#ifndef __RECOGNIZE_TEXT_H
#define __RECOGNIZE_TEXT_H

#include <vector>
#include <set>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "er_filter.hpp"
#include "er_formwords.hpp"


using namespace cv;
using namespace std;

namespace recognize_text
{
	void recognize_text(Mat &src);
	void words_draw(Mat &img, list<set<er::ERStat> >& words);
	void components_draw(Mat &src, vector<er::ERStat> &comps);
	void groups_draw(Mat &src, vector<Rect> &groups);
	void er_show(vector<Mat> &channels, vector<vector<er::ERStat> > &regions);
}

#endif
