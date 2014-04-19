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

Mat wlDraw(ERset &triplet, ERWordLine &wl)
{
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

	// Get the word boundary points
	it = triplet.begin();
	Point TL = Point( it->rect.tl().x, it->rect.tl().y );
	it = triplet.end();
	it--;
	Point TR = Point( it->rect.tl().x, it->rect.tl().y );
	TR.x = TR.x + it->rect.width;

	double b;
	for (int i=0; i<4; i++)
	{
		switch (i)
		{
			case 0: b = wl.tau.t[0]; break;
			case 1: b = wl.tau.t[1]; break;
			case 2: b = wl.tau.b[0]; break;
			default: b = wl.tau.b[1]; break;
		}
		Point p1 = Point(TL.x, (wl.tau.slope*TL.x)+b);
		Point p2 = Point(TR.x, (wl.tau.slope*TR.x)+b);

		line(mask, p1, p2, Scalar(255), 1 );
	}

	return mask;

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

double wl_dist(ERWordLine& wl1, ERWordLine& wl2)
{

	wl_params& tau1 = wl1.tau;
	wl_params& tau2 = wl2.tau;

	double h = std::max(tau1.h, tau2.h);
	double xL = std::min(tau1.x_min, tau2.x_min);
	double xR = std::max(tau1.x_max, tau2.x_max);

	double min_t = numeric_limits<double>::max();
	double min_b = numeric_limits<double>::max();
	for (int i=0; i<2; i++)
	{
		for (int j=0; j<2; j++)
		{
			double tdL = std::abs((tau1.slope*xL+tau1.t[i]) - (tau2.slope*xL+tau2.t[j]));
			double tdR = std::abs((tau1.slope*xR+tau1.t[i]) - (tau2.slope*xR+tau2.t[j]));
			double tdelta = (std::max(tdL, tdR) / h);

			double bdL = std::abs((tau1.slope*xL+tau1.b[i]) - (tau2.slope*xL+tau2.b[j]));
			double bdR = std::abs((tau1.slope*xR+tau1.b[i]) - (tau2.slope*xR+tau2.b[j]));
			double bdelta = (std::max(bdL, bdR) / h);
			
			if (tdelta < min_t)
				min_t = tdelta;

			if (bdelta < min_b)
				min_b = bdelta;
		}
	}
	
	return std::max(min_t, min_b);
}

bool v3(ERset& quad)
{

	// Form triplet1 and triplet2
	ERset triplet1, triplet2;
	ERset::iterator it = quad.begin();
	//e.g. 	quad 			= ABCD
	//			triplet1 	= ABC
	//			triplet2  = BCD
	
	// A
	triplet1.insert(*it++);

	// B
	triplet1.insert(*it);
	triplet2.insert(*it++);

	// C
	triplet1.insert(*it);
	triplet2.insert(*it++);

	// D
	triplet2.insert(*it);


	ERWordLine wl1 = ERWordLine(triplet1);
	ERWordLine wl2 = ERWordLine(triplet2);

	Mat im1 = wlDraw(triplet1, wl1);
	Mat im2 = wlDraw(triplet2, wl2);

	imshow("word 1", im1);
	imshow("word 2", im2);
	double d = wl_dist(wl1, wl2);
	cout << "Distance is " << d << endl;
	waitKey();

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

					if (subset_1N.size() == 4)
					{
						if ( ! v3(subset_1N) )
							continue;
					}

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
