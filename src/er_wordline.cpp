#include "er_wordline.hpp"

using namespace cv;
using namespace std;

namespace er
{

ERWordLine::ERWordLine(const ERset &triplet)
{

	CV_Assert( triplet.size() == 3);

	// Find the max region height
	ERset::iterator it = triplet.begin();
	double h_max = 0;
	for ( ; it != triplet.end(); it++)
	{
		double h = it->rect.height;
		if (h > h_max)
			h_max = h;
	}
	tau.h = h_max;

	// Estimate 4 word lines vector < pair< a (slope), b (intercept) > >
	estimateWordLines(triplet);

	// Get the word boundary x values
	it = triplet.begin();
	tau.x_min = it->rect.tl().x;
	it = triplet.end();
	it--;
	tau.x_max = it->rect.br().x;

}


double ERWordLine::calcError1(vector<Point> pts, int idx)
{
	Mat Y = (Mat_<double>(3, 1) << pts[0].y, pts[1].y, pts[2].y);
	Mat X = (Mat_<double>(3, 1) << pts[0].x, pts[1].x, pts[2].x);

	// idx is the index of the pts vector for which the potential line intersects
	double b = pts[idx].y - ( tau.slope * pts[idx].x ); 
	Mat B = (Mat_<double>(3, 1) << b, b, b);
	
	Mat E;
	cv::pow(Y - ( tau.slope*X + B ), 2, E);
	Scalar Es = cv::sum(E);

	return Es[0];
}

double ERWordLine::calcError2(vector<Point> pts, int idx1, int idx2)
{
	Mat Y = (Mat_<double>(3, 1) << pts[0].y, pts[1].y, pts[2].y);
	Mat X = (Mat_<double>(3, 1) << pts[0].x, pts[1].x, pts[2].x);

	// idx is the index of the pts vector for which the potential line intersects
	double b1 = pts[idx1].y - ( tau.slope * pts[idx1].x ); 
	Mat B1 = (Mat_<double>(3, 1) << b1, b1, b1);
	
	double b2 = pts[idx2].y - ( tau.slope * pts[idx2].x ); 
	Mat B2 = (Mat_<double>(3, 1) << b2, b2, b2);

	Mat E1;
	cv::pow(Y - ( tau.slope*X + B1 ), 2, E1 );
	Mat E2;
	cv::pow(Y - ( tau.slope*X + B2 ), 2, E2 );
	Mat E = cv::min(E1,E2);
	Scalar Es = cv::sum(E);

	return Es[0];
}

pair<double,double> ERWordLine::fitLines(vector<Point> pts)
{
	// Only fit lines for triplets
	CV_Assert( pts.size() == 3 );

	// Fit the first line
	double min_error = numeric_limits<double>::max();
	double min_idx_1 = 0;
	for (int i=0; i<(int)pts.size(); i++)
	{
			double error = calcError1(pts, i);
			if (error < min_error)
			{
				min_error = error;
				min_idx_1 = i;
			}
	}

	// Fit the second line
	min_error = numeric_limits<double>::max();
	double min_idx_2 = 0;
	for (int i=0; i<(int)pts.size(); i++)
	{
		if (i == min_idx_1)
			continue;

		double error = calcError2(pts, min_idx_1, i);
		if (error < min_error)
		{
			min_error = error;
			min_idx_2 = i;
		}
	}
	
	double b1 = pts[min_idx_1].y - ( tau.slope * pts[min_idx_1].x );
	double b2 = pts[min_idx_2].y - ( tau.slope * pts[min_idx_2].x );
	pair<double,double> line = make_pair(b1, b2);

	return line;
}

double ERWordLine::median(vector<double> scores)
{
  size_t size = scores.size();
  sort(scores.begin(), scores.end());

  double median;
  if (size  % 2 == 0)
      median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
  else 
      median = scores[size / 2];

  return median;
}

void ERWordLine::LeastMedSquaresDirection(vector<Point> pts)
{

	double a_best = 0;
	double b_best = 0;
	double d_min = numeric_limits<double>::max();

	unsigned int combs[6][3] =
	{
		{2, 1, 0},
		{2, 0, 1},
		{1, 2, 0},
		{1, 0, 2},
		{0, 2, 1},
		{0, 1, 2}
	};

	for (int i=0; i<6; i++)
	{

		// Choose new x and y combination
		double xi,xj,xk,yi,yj,yk;
		xi = (double)pts[combs[i][0]].x;
		xj = (double)pts[combs[i][1]].x;
		xk = (double)pts[combs[i][2]].x;
		yi = (double)pts[combs[i][0]].y;
		yj = (double)pts[combs[i][1]].y;
		yk = (double)pts[combs[i][2]].y;

		// Set up X and y matricies
		Mat X = (Mat_<double>(3,2) << 1, xi, 1, xj, 1, xk);
		Mat y = (Mat_<double>(3,1) << yi, yj, yk);

		// Calculate slope and intercept for this permutation
		double b = (yi - yk)/(xi - xk);
		double a = (yj + yk - b*(xj + xk) )/2;
		
		// Calculate residuals
		Mat w = Mat::zeros(2,1, CV_64F);		
		w.at<double>(0,0) = a;
		w.at<double>(1,0) = b;
		Mat r = cv::abs(y - X*w);

		// Find median
		vector<double> findmymed(3);
		findmymed[0] = r.at<double>(0,0);
		findmymed[1] = r.at<double>(1,0);
		findmymed[2] = r.at<double>(2,0);
		double d = median(findmymed);
	
		// Update best median residual
		if (d < d_min)
		{
			d_min = d;
			a_best = a;
			b_best = b;
		}

	}
	tau.slope = b_best;
}

void ERWordLine::estimateWordLines(const ERset &triplet)
{
	vector<Point> top_pts, bot_pts;
	for (ERset::iterator it=triplet.begin(); it != triplet.end(); it++)
	{
		// Top-left corner
		top_pts.push_back(it->rect.tl());

		// Bottom-left corner
		Point b = it->rect.br();
		b.x = b.x + it->rect.width;
		bot_pts.push_back(b);
	}

	// Use the bottom points to determine word line direction
	LeastMedSquaresDirection(bot_pts);

	// Using the word line direction, fit two distince lines to the
	// remaining top and bottom lines to find the four line estimates
	pair<double,double> top_intercepts = fitLines(top_pts);
	pair<double,double> bot_intercepts = fitLines(bot_pts); 

	// Assign y-intercept values to the four word lines
	tau.t[0] = top_intercepts.first;
	tau.t[1] = top_intercepts.second;
	tau.b[0] = bot_intercepts.first;
	tau.b[1] = bot_intercepts.second;


}



} //namespace er
