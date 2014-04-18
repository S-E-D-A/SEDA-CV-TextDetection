#ifndef __ER_FORMWORDS_H
#define __ER_FORMWORDS_H

#include "er.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> //TODO: remove

#include <list>
#include <algorithm>
#include <set>
#include <cmath>
#include <iostream>
#include <vector>

namespace er
{

void erFormWords(cv::Mat &img, std::vector<cv::Mat> &channels, std::vector<std::vector<er::ERStat> > &regions);

}

#endif
