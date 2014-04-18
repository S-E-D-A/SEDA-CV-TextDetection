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


//struct stat_x_cmp 
//{
//	bool operator() (const ERStat erc1, const ERStat erc2)
//	{
//		return (erc1.rect.tl().x < erc2.rect.tl().x);
//	}
//};

// Set of ERs. Sorted by x position of top-left corner of bounding box
//typedef set<ERStat, stat_x_cmp> ERset;
typedef set<ERStat> ERset;

//struct ERWord
//{
//	vector<ERStat> letters;
//};

//bool sortByChannel(ERChar erc1, ERChar erc2)
//{
//	return (erc1.channel > erc2.channel);
//}
//bool sortByX(Ptr<ERChar> erc1, Ptr<ERChar> erc2)
//{
//		return (erc1->stat.rect.tl().x < erc2->stat.rect.tl().x);
//}


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
//
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
		//const ERChar* er1_ptr = (*it1)Ptr<ERChar>();
		//const ERChar* er1_ptr = (*it1)(ERChar*());
		//const ERChar* er2_ptr = (*it2)Ptr<ERChar>();
		//if ( er1_ptr != er2_ptr)
		if ( !((*it1) == (*it2)) )
			return false;
		it1++;
		it2++;
	}

	return true;

}

//vector<vector<int> > GenerateCombinations(int n, int k)
//{
//
//	vector<vector<int> > all_combs;
//	vector<int> comb;
//
//	vector<bool> v(n);
//  fill(v.begin() + n - k, v.end(), true);
//
//  do {
//      for (int i = 0; i < n; ++i) {
//          if (v[i]) {
//							comb.push_back(i+1);
//              //std::cout << (i+1) << " ";
//          }
//      }
//      //std::cout << "\n";
//			all_combs.push_back(comb);
//			comb.clear();
//  } while (std::next_permutation(v.begin(), v.end()));
//
//
//	return all_combs;
//}
//
//double median(Mat a, int N)
//{
//	Mat s;
//	cv::sort(a,s, CV_SORT_EVERY_ROW);
//
//	double med;
//	if (N%2)
//	{
//		int middle1 = N/2;
//		int middle2 = middle1 + 1;
//		double med1 = s.at<double>(middle1);
//		double med2 = s.at<double>(middle2);
//		med = (med1 + med2)/2;
//	}
//	else
//	{	
//		int middle = N/2;
//		med = s.at<double>(middle);
//	}
//
//	return med;
//
//}
//
//Point LeastMedSquares(vector<Point> pts)
//{
//
//	int N = (int)pts.size();
//
//	Mat X = Mat::ones(N,2, CV_64F);		
//	for (int n=0; n<N; n++)
//		X.at<double>(n,1) = pts[n].x;
//
//	Mat y = Mat::zeros(N,1, CV_64F);
//	for (int n=0; n<N; n++)
//		y.at<double>(n,0) = pts[n].y;
//
//	vector<vector<int> > combs = GenerateCombinations(N, 3);
//
//	double a_best = 0;
//	double b_best = 0;
//	double d_min = 9999999;
//
//	for (int i=0; i<(int)combs.size(); i++)
//	{
//		vector<int> idx;
//		idx = combs[i];
//
//		double xi,xj,xk,yi,yj,yk;
//		xi = pts[idx[0]].x;
//		xj = pts[idx[1]].x;
//		xk = pts[idx[2]].x;
//		yi = pts[idx[0]].y;
//		yj = pts[idx[1]].y;
//		yk = pts[idx[2]].y;
//
//		double b = (yi - yk)/(xi - xk);
//		double a = (yj + yk - b*(xj + xk) )/2;
//		
//		Mat w = Mat::zeros(2,1, CV_64F);		
//		w.at<double>(0,0) = a;
//		w.at<double>(1,0) = b;
//		Mat r = cv::abs(y - X*w);
//		double d = median(r,N);
//
//		if (d < d_min)
//		{
//			d_min = d;
//			a_best = a;
//			b_best = b;
//		}
//
//	}
//
//	Point out;
//	out.x = a_best;
//	out.y = b_best;
//
//	return out;
//}

void erShow(ERset &er_set, double delay)
{

	// Increment er_set iterator to first non-root ER, since the root ER may be a dummy ER
	ERset::iterator it = er_set.begin();

	// Grab the first ER and use its image pointer to initialize the size of a
	// zeros image which will be used to display all the ERs in er_set
	Mat mask = Mat::zeros((*it).im_ptr->rows+2, (*it).im_ptr->cols+2, CV_8UC1);

	//vector<Point> pts;
	//Point leftmost, rightmost;
	//leftmost.x = cols;
	//rightmost.x = 0;
	for ( ; it != er_set.end(); it++)
	{
		ERStat er = (*it);
		//int c = (*it)->channel;
		if (er.parent != NULL)
		{
			int newmaskval = 255;
			int flags = 4 + (newmaskval << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
			Mat im = (*er.im_ptr);
			floodFill(im,mask,Point(er.pixel%im.cols, er.pixel/im.cols), Scalar(255),0,Scalar(er.level),Scalar(0),flags);
			Point BR = er.rect.br();
			BR.x = BR.x + 1; BR.y = BR.y + 1; //To account for mask size difference
			//circle(masks[c], BR, 5, Scalar(255));
			//pts.push_back(BR);

			//if (BR.x < leftmost.x)
			//	leftmost.x = BR.x;
			//if (BR.x > rightmost.x)
			//	rightmost.x = BR.x;
		}
	}
	//Point lineParams = LeastMedSquares(pts);
	//leftmost.y = lineParams.x + lineParams.y*leftmost.x;
	//rightmost.y = lineParams.x + lineParams.y*rightmost.x;
	//line(masks[0], leftmost, rightmost, Scalar(255), 2);

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


	// Compare angle between centroids
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

void erFormWords(vector<vector<ERStat> > &regions)
{

	CV_Assert( !regions.empty() );

	ERset er_all;
	for (int i=0; i<(int)regions.size(); i++)
	{
		for (int j=0; j<(int)regions[i].size(); j++)
		{
			//Ptr<ERStat> er = new
			er_all.insert(regions[i][j]);
		}
	}

	// Show all regions
	erShow(er_all, 0);

	vector<list<ERset> > words;
	// Vector of list of sets to store candidate words
	// Outer vector:	determines length
	// 								[0] all length 1 words or subwords
	// 								[1] all length 2 words or subwords
	// 																:
	//																 
	// Inner list:	words or subwords of that length

	list<ERset> words_of_length;
	// Create pairwise words
	{
		ERset::iterator it1, it2;
		it1 = er_all.begin();
		while ( it1 != er_all.end() ) 
		{
			if ((*it1).parent == NULL)
			{
				cout << "it1 null parent " << endl;
				it1++;
			}
			it2 = it1;
			it2++;

			ERset pair;
			while ( it2 != er_all.end() )
			{
				ERStat er1 = *it1;
				ERStat er2 = *it2;
				if ( v1(er1, er2) )
				{
					pair.insert(er1);
					pair.insert(er2);
					words_of_length.push_back(pair);
				}
				pair.clear();
				it2++;
			}
			it1++;
		}
	}
	words.push_back(words_of_length);
	words_of_length.clear();
	

	// Create words of length N > 2
	for (int d = 1; !words[d-1].empty(); d++ )
	{
		list<ERset>::iterator it1, it2;
		it1	= words[d-1].begin();	
		for (it1 = words[d-1].begin(); it1 != words[d-1].end(); it1++ )
		{
			// Word or subword at it1
			ERset er_set1 = *it1;
				
			// First letter of word or subword at it1
			ERStat subset_1 = (*er_set1.begin());

			// Generate the subword from n=2,...,N at it1
			ERset subset_2N;
			ERset::iterator it_mid_1;
			it_mid_1 = er_set1.begin();
			it_mid_1++;
			for ( ; it_mid_1 != er_set1.end(); it_mid_1++)	
			{
				subset_2N.insert(*it_mid_1);
			}

			// New word or subword to be added	
			ERset subset_1N;
			for (it2 = it1; it2 != words[d-1].end(); it2++ )
			{
				// Word or subword at it2
				ERset er_set2 = *it2;

				// Generate subword from n=1,...,N-1 at it2
				ERset subset_1N1;
				ERset::iterator it_mid_2, it_end_2;
				it_mid_2 = er_set2.begin();
				it_end_2 = er_set2.end();
				it_end_2--; //Stopping point at N-2
				for ( ; it_mid_2 != it_end_2; it_mid_2++)
				{
					subset_1N1.insert(*it_mid_2);
				}

				// Compare to word at it2 which should be length N-1
				CV_Assert( subset_2N.size() == subset_1N1.size() );

				// Last letter of word of subword at it2
				ERStat subset_N;
				ERset::iterator it_last_2 = er_set2.end();
				it_last_2--;
				subset_N = (*it_last_2);

				if ( compareERStat_sets(subset_2N, subset_1N1) )
				{
					// Insert the first letter from it1
					subset_1N.insert(subset_1);
					
					// Insert all the letters from the overlap of it1 and it2 (length N-1)
					ERset::iterator it_mid;
					it_mid = subset_1N1.begin();
					for ( ; it_mid != subset_1N1.end(); it_mid++)
						subset_1N.insert(*it_mid);

					// Insert the last letter from it2
					subset_1N.insert(subset_N);


					words_of_length.push_back(subset_1N);
				}
				subset_1N.clear();
			}
		}

		words.push_back(words_of_length);
		words_of_length.clear();


		cout << "Just finished words of length " << d+1 << endl;
		
	}

	// Show all ERs
	//for (int d=0; d<(int)words.size(); d++)
	//		for (int s=0; s<(int)words[d].size(); s++)
	//			erShow(words[d][s], MS_DELAY);

	// Prune subwords from words
	pruneSubwords(words);

	// Show all ERs
	for (int d=0; d<(int)words.size(); d++)
	{
		list<ERset>::iterator s;
		for (s=words[d].begin(); s != words[d].end(); s++)
			erShow((*s), 0);
	}

}

} //end namespace er
