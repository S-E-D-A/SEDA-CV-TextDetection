#include "er_formwords.hpp"

using namespace cv;
using namespace std;

#define PI 3.14159265


namespace er
{

const double DIST_MAX_RATIO = 2;
const double DIST_MIN_RATIO = 0.2;
const double HEIGHT_RATIO = 2;
const double HORIZ_ANGLE = 20;
const double MS_DELAY = 50;
const int MIN_WORD_LENGTH = 4;


typedef set<ERStat> ERset;

//bool isParentChild_helper(ERStat* node, ERStat* compare, bool look_up)
//{
//
//	if (node == compare)
//		return true;
//
//	if (node == NULL)
//		return false;
//
//	if (look_up)
//		return ( isParentChild_helper(node->parent, compare, true) );
//	else
//	{
//		for (ERStat * child = node->child; child; child = child->next)
//   	{
//    	if (isParentChild_helper( child, compare, false ) )
//				return true;
//    }
//	}
//	return false;
//
//}

//bool isParentChild(Ptr<ERChar> erc1, Ptr<ERChar> erc2)
//{
//	// Must be formed from same channel to be paren-child
//	if (erc1->channel != erc2->channel)
//		return false;
//
//	//TODO: Will the references to these objects be the same as the parents & childs
//	//			in the ERStat* tree? Have not tested yet
//	stat1 = erc1->stat;
//	stat2 = erc2->stat;
//
//	// If nodes are the same
//	if (erc1 == erc2)
//		return true;
//
//	if (isParentChild_helper(erc1, erc2, true) || isParentChild_helper(erc1, erc2, false) )
//		return true
//
//	return false;
//
//}

bool compareERStat_sets(ERset s1, ERset s2)
{
	ERset::iterator it1, it2;

	it1 = s1.begin();
	it2 = s2.begin();
	while ( it1 != s1.end() && it2 != s2.end() )
	{
		if ( !((*it1) == (*it2)) )
			return false;
		it1++;
		it2++;
	}

	return true;

}

//vector<vector<int> > GenerateIndexPermutations()
//{
//	vector<int> myints = {0, 1, 2};
//  sort (myints.begin(),myints.end());
//
//	vector< vector<int> > perms;
//  //cout << "The 3! possible permutations with 3 elements:\n";
//  do {
//    //cout << myints[0] << ' ' << myints[1] << ' ' << myints[2] << '\n';
//		perms.push_back(myints);
//  } while ( next_permutation(myints.begin(),myints.end()) );
//
//  //cout << "After loop: " << myints[0] << ' ' << myints[1] << ' ' << myints[2] << '\n';
//	return perms;
//}

double calcError1(vector<Point> pts, int idx, double slope)
{
	Mat Y = (Mat_<double>(3, 1) << pts[0].y, pts[1].y, pts[2].y);
	Mat X = (Mat_<double>(3, 1) << pts[0].x, pts[1].x, pts[2].x);

	// idx is the index of the pts vector for which the potential line intersects
	double b = pts[idx].y - ( slope * pts[idx].x ); 
	Mat B = (Mat_<double>(3, 1) << b, b, b);
	
	Mat E;
	cv::pow(Y - ( slope*X + B ), 2, E);
	Scalar Es = cv::sum(E);
	double error = Es[0]; 
	cout << "Error at idx " << idx << " is " << error << endl;

	return error;
}

double calcError2(vector<Point> pts, int idx1, int idx2,  double slope)
{
	Mat Y = (Mat_<double>(3, 1) << pts[0].y, pts[1].y, pts[2].y);
	Mat X = (Mat_<double>(3, 1) << pts[0].x, pts[1].x, pts[2].x);

	// idx is the index of the pts vector for which the potential line intersects
	double b1 = pts[idx1].y - ( slope * pts[idx1].x ); 
	Mat B1 = (Mat_<double>(3, 1) << b1, b1, b1);
	
	double b2 = pts[idx2].y - ( slope * pts[idx2].x ); 
	Mat B2 = (Mat_<double>(3, 1) << b2, b2, b2);

	Mat E1;
	cv::pow(Y - ( slope*X + B1 ), 2, E1 );
	Mat E2;
	cv::pow(Y - ( slope*X + B2 ), 2, E2 );
	Mat E = cv::min(E1,E2);
	Scalar Es = cv::sum(E);
	double error = Es[0]; 

	cout << "E1 is " << E1 << endl;
	cout << "E2 is " << E2 << endl;
	cout << "E is " << E << endl;
	cout << "Error is " << error << endl;

	return error;
}

pair<double,double> fitLines(vector<Point> pts, double slope)
{
	// Only fit lines for triplets
	CV_Assert( pts.size() == 3 );

	// Fit the first line
	double min_error = numeric_limits<double>::max();
	double min_idx_1 = 0;
	for (int i=0; i<(int)pts.size(); i++)
	{
			double error = calcError1(pts, i, slope);
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

		double error = calcError2(pts, min_idx_1, i, slope);
		if (error < min_error)
		{
			min_error = error;
			min_idx_2 = i;
		}
	}
	
	double b1 = pts[min_idx_1].y - ( slope * pts[min_idx_1].x );
	double b2 = pts[min_idx_2].y - ( slope * pts[min_idx_2].x );
	pair<double,double> line = make_pair(b1, b2);

	return line;
}

double median(vector<double> scores)
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

double LeastMedSquaresDirection(vector<Point> pts)
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
		cout << "R is " << r << endl;

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
	cout << "a_best is " << a_best << endl;
	cout << "b_best is " << b_best << endl;
	return b_best;
}

vector<pair<double,double> > estimateWordLines(ERset triplet)
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
	double slope = LeastMedSquaresDirection(bot_pts);

	// Using the word line direction, fit two distince lines to the
	// remaining top and bottom lines to find the four line estimates
	pair<double,double> top_intercepts = fitLines(top_pts, slope);
	pair<double,double> bot_intercepts = fitLines(bot_pts, slope); 

	vector<pair<double,double> > lines;
	lines.push_back( make_pair(slope, top_intercepts.first) );
	lines.push_back( make_pair(slope, top_intercepts.second) );
	lines.push_back( make_pair(slope, bot_intercepts.first) );
	lines.push_back( make_pair(slope, bot_intercepts.second) );

	return lines;
}

bool v3(ERset& triplet)
{
	CV_Assert( triplet.size() == 3);

	// Draw the triplet
	ERset::iterator it = triplet.begin();
	int cols = (*it).im_ptr->cols;
	int rows = (*it).im_ptr->rows;
	Mat mask = Mat::zeros(rows+2, cols+2, CV_8UC1);
	for ( ; it != triplet.end(); it++)
	{
		ERStat er = (*it);
		int newmaskval = 255;
		int flags = 4 + (newmaskval << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
		Mat im = (*er.im_ptr);
		floodFill(im,mask,Point(er.pixel%im.cols, er.pixel/im.cols), Scalar(255),0,Scalar(er.level),Scalar(0),flags);
	}

	// Estimate 4 word lines vector < pair< a (slope), b (intercept) > >
	vector<pair<double,double> > wordlines = estimateWordLines(triplet);

	// Get the word boundary x values
	it = triplet.begin();
	int x_min = it->rect.tl().x;
	it = triplet.end();
	it--;
	int x_max = it->rect.br().x;

	// Draw the word lines
	double slope = wordlines[0].first;
	for (int i=0; i<(int)wordlines.size(); i++)
	{
		double b = wordlines[i].second;
		Point pp1 = Point(x_min, (slope*x_min)+b);
		Point pp2 = Point(x_max, (slope*x_max)+b);

		line(mask, pp1, pp2, Scalar(255), 1 );
	}

	cout << "drew lines" << endl;
	imshow("win", mask);
	waitKey();
	
	return true;

}

void erShow(ERset &er_set, double delay)
{

	// Increment er_set iterator to first non-root ER, since the root ER may be a dummy ER
	ERset::iterator it = er_set.begin();

	int cols = (*it).im_ptr->cols;
	int rows = (*it).im_ptr->rows;

	// Grab the first ER and use its image pointer to initialize the size of a
	// zeros image which will be used to display all the ERs in er_set
	Mat mask = Mat::zeros(rows+2, cols+2, CV_8UC1);

	for ( ; it != er_set.end(); it++)
	{
		ERStat er = (*it);
		if (er.parent != NULL)
		{
			int newmaskval = 255;
			int flags = 4 + (newmaskval << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
			Mat im = (*er.im_ptr);
			floodFill(im,mask,Point(er.pixel%im.cols, er.pixel/im.cols), Scalar(255),0,Scalar(er.level),Scalar(0),flags);
			//Point BR = er.rect.br();
			//BR.x = BR.x + 1; BR.y = BR.y + 1; //To account for mask size difference
			//circle(mask, BR, 5, Scalar(255));

		}
	}

	imshow("Regions", mask);
	cvMoveWindow("Regions", 200, 50);
	waitKey(delay);

}	



bool v1(const ERStat& er1, const ERStat& er2)
{
	const Rect& r1 = er1.rect;
	const Rect& r2 = er2.rect;

	// Compare centroid distances
	double w_max;
	if (r1.width > r2.width)
		w_max = r1.width;
	else
		w_max = r2.width;

	Point c1;
	c1.x = r1.x + (r1.width/2);
	c1.y = r1.y + (r1.height/2);

	Point c2;
	c2.x = r2.x + (r2.width/2);
	c2.y = r2.y + (r2.height/2);


	// Hueristic for parent-child relationship
	double d = norm(c2-c1);
	if (d > w_max*DIST_MAX_RATIO || d < w_max*DIST_MIN_RATIO)
		return false;


	// Compare angle between centroids TODO: use moment[0] not center of BB
	Point diag = c2 - c1;
	double deg = asin(diag.y / norm(diag)) * 180.0 / PI;

	if (std::abs(deg) > HORIZ_ANGLE)
		return false;
	

	// Compare heights
	double h1 = r1.height;
	double h2 = r2.height;

	if (h1/h2 > HEIGHT_RATIO || h2/h1 > HEIGHT_RATIO)
		return false;

	return true;
}

// Assume s1 is shorter than s2
bool isSubWord(ERset s1, ERset s2)
{
	if (s1.empty() || s2.empty())
		return false;

	ERset::iterator it1 = s1.begin();
	ERset::iterator it2 = s2.begin();
	
	while ( it1 != s1.end() && it2 != s2.end() )
	{
		if ( (*it1) == (*it2) )
		{
			it1++;
			it2++;
		}
		else
			it2++;
	}

	if ( it1 == s1.end())
		return true;
	else
		return false;
}

void pruneSubwords(vector<list<ERset> > &words)
{

	for (int d=0; d<(int)words.size(); d++)
	{

		// Clear minimum word length words
		if ( d < MIN_WORD_LENGTH-2 )
		{
			words[d].clear();
			continue;
		}

		list<ERset>::iterator it1;
		list<ERset>::iterator it2;

		bool erased = true;
		for (it1 = words[d-1].begin(); it1 != words[d-1].end(); )
		{

			ERset small = (*it1);
			for(it2 = words[d].begin(); it2 != words[d].end(); it2++)
			{
				ERset big = (*it2);
				if ( isSubWord(small,big) )
				{
					it1 = words[d-1].erase(it1);
					erased = true;
					break;
				}
			}

			if (!erased)
				it1++;
			erased = false;
		}
	}
}

// begin: iterator at FIRST er in subword
// end:		iterator at LAST+1 er in subword
ERset getSubWord(ERset::iterator begin, unsigned int length)
{
	ERset subword;
	//ERset::iterator stop = ++end;
	//while ( begin != stop )
	for (unsigned int i=0; i<length; i++)
	{
		subword.insert(*begin);
		begin++;
	}
	return subword;
}

void erFormWords(set<ERStat> &regions)
{

	CV_Assert( !regions.empty() );

	// Show all regions
	erShow(regions, 0);

	vector<list<ERset> > words;
	// Vector of list of sets to store candidate words
	// Outer vector:	determines length
	// 								[0] all length 1 words or subwords
	// 								[1] all length 2 words or subwords
	// 																:
	//																 
	// Inner list:	words or subwords of that length


	// --- Create ER sequences of length 2 ---
	//
	//		Runtime: O(n^2)?, n is number of ERs
	list<ERset> all_pairs;
	{
		ERset::iterator it1, it2;
		it1 = regions.begin();
		while ( it1 != regions.end() ) 
		{
			if ((*it1).parent == NULL)
			{
				cout << "it1 null parent " << endl;
				it1++;
			}
			it2 = it1;
			it2++;

			ERset pair;
			while ( it2 != regions.end() )
			{
				ERStat er1 = *it1;
				ERStat er2 = *it2;
				if ( v1(er1, er2) )
				{
					pair.insert(er1);
					pair.insert(er2);
					all_pairs.push_back(pair);
				}
				pair.clear();
				it2++;
			}
			it1++;
		}
	}
	words.push_back(all_pairs);
	all_pairs.clear();
	
	// --- Create ER sequences of length 3 ---
	list<ERset> all_words_of_length;
	for (int d = 0; !words[d].empty() && d < 7; d++ )
	{
		list<ERset>::iterator it1, it2;
		it1	= words[d].begin();	
		for (it1 = words[d].begin(); it1 != words[d].end(); it1++ )
		{
			// Entire word or subword at it1
			// e.g.		ABCDE
			ERset er_set1 = *it1;
				
			// First letter of word or subword at it1
			// e.g.		A
			ERStat first_letter = *getSubWord(er_set1.begin(), 1).begin();

			// Generate the subword from n = 2, ..., N at it1
			// e.g.		BCDE
			ERset::iterator mid = er_set1.begin();
			ERset middle_letters_2_N = getSubWord(++mid, er_set1.size()-1);
	
			// New word or subword to be added	
			// e.g. ABCDEF (empty for now)
			ERset subset_1N;
			for (it2 = it1; it2 != words[d].end(); it2++ )
			{
				// Word or subword at it2
				// e.g. BCDEF
				ERset er_set2 = *it2;

				// Generate subword from n=1,...,N-1 at it2
				// e.g. BCDE
				ERset middle_letters_1_N1 = getSubWord(er_set2.begin(), er_set2.size()-1);

				// Compare to word at it2 which should be length N-1
				CV_Assert( middle_letters_2_N.size() == middle_letters_1_N1.size() );

				// Last letter of word of subword at it2
				// e.g. F
				ERStat subset_N;
				ERset::iterator it_last_2 = er_set2.end();
				it_last_2--;
				subset_N = (*it_last_2);

				if ( compareERStat_sets(middle_letters_2_N, middle_letters_1_N1) )
				{
					// Insert the first letter from it1
					// e.g. A
					subset_1N.insert(first_letter);
					
					// Insert all the letters from the overlap of it1 and it2 (length N-1)
					// e.g. BCDE
					ERset::iterator it_mid;
					it_mid = middle_letters_1_N1.begin();
					for ( ; it_mid != middle_letters_1_N1.end(); it_mid++)
						subset_1N.insert(*it_mid);

					// Insert the last letter from it2
					// eg. F
					subset_1N.insert(subset_N);

					if (subset_1N.size() == 3)
						v3(subset_1N);

					all_words_of_length.push_back(subset_1N);
				}
				subset_1N.clear();
			}
		}

		words.push_back(all_words_of_length);
		all_words_of_length.clear();

		cout << "Finished words of length " << d+1 << endl;
		
	}

	// Prune subwords from words
	pruneSubwords(words);

	// Show all ERs
	//for (int d=0; d<(int)words.size(); d++)
	//{
	//	list<ERset>::iterator s;
	//	for (s=words[d].begin(); s != words[d].end(); s++)
	//		erShow((*s), 10);
	//}

}

} //namespace er
