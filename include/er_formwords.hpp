#ifndef __ER_FORMWORDS_H
#define __ER_FORMWORDS_H

#include "er.hpp"
#include "er_wordline.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> //TODO: remove

#include <list>
#include <algorithm>
#include <utility>
#include <limits>
#include <set>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <vector>

namespace er
{

void erFormWords(std::set<er::ERStat> &regions);

}

#endif
