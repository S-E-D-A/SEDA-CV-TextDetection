#ifndef __ER_WORDLINE_H
#define __ER_WORDLINE_H

#include "er.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <algorithm>
#include <utility>
#include <limits>
#include <set>
#include <cmath>
#include <vector>
#include <iostream>

namespace er
{

typedef std::set<ERStat> ERset;

struct wl_params
{
	double slope;
	double t[2];
	double b[2];
	//double t1;
	//double t2;
	//double b1;
	//double b2;
	double x_min;
	double x_max;
	double h;
};

class ERWordLine
{
	public:
		ERWordLine(const ERset &triplet);
		wl_params tau;

	private:
		
		double calcError1(std::vector<cv::Point> pts, int idx);
		double calcError2(std::vector<cv::Point> pts, int idx1, int idx2);
		std::pair<double,double> fitLines(std::vector<cv::Point> pts);
		double median(std::vector<double> scores);
		void LeastMedSquaresDirection(std::vector<cv::Point> pts);
		void estimateWordLines(const ERset &triplet);
};




}

#endif
