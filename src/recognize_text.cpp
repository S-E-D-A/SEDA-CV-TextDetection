#include "recognize_text.hpp"

using namespace er;

namespace recognize_text
{

	void recognize_text(Mat &src)
	{	

		// Extract channels to be processed individually
		vector<Mat> channels;
		computeNMChannels(src, channels);

		// Use subset of channels to simplify
		//vector<Mat> channels; //TODO: remove
		//channels.push_back(channels_all[0]);
		//channels.push_back(255-channels[0]);

		int cn = (int)channels.size();
		// Append negative channels to detect ER- (bright regions over dark background)
		for (int c = 0; c < cn-1; c++)
			channels.push_back(255-channels[c]);

		// Create ERFilter objects with the 1st and 2nd stage default classifiers
		Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("models/trained_classifierNM1.xml"),16,0.00015f,0.13f,0.2f,true,0.1f);
		Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("models/trained_classifierNM2.xml"),0.5);

		vector<vector<ERStat> > regions(channels.size());
		// Apply the default cascade classifier to each independent channel (could be done in parallel)
		cout << "Extracting Class Specific Extremal Regions from " << (int)channels.size() << " channels ..." << endl;
		cout << "    (...) this may take a while (...)" << endl << endl;
		for (int c=0; c<(int)channels.size(); c++)
		{
			er_filter1->run(channels[c], regions[c]);
			er_filter2->run(channels[c], regions[c]);
		}

		
		set<ERStat> all_regions;
		for (int i=0; i<(int)regions.size(); i++)
			for (int j=0; j<(int)regions[i].size(); j++)
				all_regions.insert(regions[i][j]);

		list<ERset> out;
		erFormWords(out, all_regions);

		Mat im_out1 = src.clone();
		words_draw(im_out1, out);
		cout << "Done!" << endl << endl;
		imshow("Detections", im_out1);

		Mat im_out2 = src.clone();
		er_nms(out, 0.2);
		words_draw(im_out2, out);
		imshow("After NMS", im_out2);
		waitKey();

		//if( waitKey (-1) == 101)
		//	er_show(channels,all_regions);

		// memory clean-up
		er_filter1.release();
		er_filter2.release();
		regions.clear();

	}

	/* ------------------------------------------------------------------------------------*/
	/* -------------------------------- ER Recognition Algorithm --------------------------*/
	/* ------------------------------------------------------------------------------------*/

	vector<Point> getBB(set<ERStat> &region)
	{

		set<ERStat>::iterator first = region.begin();
		Point TL = first->rect.tl();
		Point BL = Point(first->rect.tl().x, first->rect.tl().y + first->rect.height);

		set<ERStat>::iterator last = region.end();
		last--;
		Point TR = Point(last->rect.br().x, last->rect.br().y - first->rect.height);
		Point BR = last->rect.br();

		vector<Point> out;
		out.push_back(TL);
		out.push_back(BL);
		out.push_back(TR);
		out.push_back(BR);

		return out;
	}

//   Sort Regions by Probability Score (after recognition)
//   **For now, sort by area
	bool sort_regions(set<ERStat> &r1, set<ERStat> &r2)
	{
		vector<Point> BBa = getBB(r1);
		vector<Point> BBb = getBB(r2);

		double a1 = abs(BBa[0].x - BBa[2].x) * abs(BBa[0].y - BBa[2].y);
		double a2 = abs(BBb[0].x - BBb[2].x) * abs(BBb[0].y - BBb[2].y);

		return (a1 > a2);
	}

	void er_nms(list<set<ERStat> > &regions, double threshold)
	{

		if (regions.empty())
			return;
		
		// Sort regions by recognition probability
		// (For now, sort by area)
		regions.sort(sort_regions);

		for (list<set<ERStat> >::iterator it_a = regions.begin(); it_a != regions.end(); it_a++)
		{
			for(list<set<ERStat> >::iterator it_b = it_a; it_b != regions.end(); it_b++)
			{
				if (it_b == it_a)
					continue;

				vector<Point> BBa = getBB(*it_a);
				vector<Point> BBb = getBB(*it_b);

				int xx1 = max(BBa[0].x, BBb[0].x);
				int yy1 = max(BBa[0].y, BBb[0].y);
				int xx2 = max(BBa[2].x, BBb[0].x);
				int yy2 = max(BBa[2].y, BBb[0].y);
				double w = xx2 - xx1;
				double h = yy2 - yy1;

				// if the bounding boxes overlap
				if ( w > 0 && h > 0 )
				{

					// calculate ratio of overlap box to it_b box
					Size s = BBb[0] - BBb[2];
					double o = (double)(w * h) / (double)(s.width * s.height); 

					// if the overlap is greater than threshold, remove it
					if (std::abs(o) > threshold)
					{
						it_b = regions.erase(it_b);
						it_b--;
					}

				}
			}
		}
		
	}

	void words_draw(Mat& img, list<set<ERStat> > & words)
	{
		for (list<ERset>::iterator it=words.begin(); it != words.end(); it++)
		{
				set<ERStat>::iterator first = it->begin();
				Point TL = first->rect.tl();
				Point BL = Point(first->rect.tl().x, first->rect.tl().y + first->rect.height);

				set<ERStat>::iterator last = it->end();
				last--;
				Point TR = Point(last->rect.br().x, last->rect.br().y - first->rect.height);
				Point BR = last->rect.br();

				Scalar c = Scalar(255,0,0);
				int t = 2;
				line(img, TL, BL, c, t); 
				line(img, BL, BR, c, t); 
				line(img, BR, TR, c, t); 
				line(img, TR, TL, c, t); 
		}
	}

	void components_draw(Mat &src, vector<ERStat> &comps)
	{
		for (int i=0; i<(int)comps.size(); i++)
		{
			if (src.type() == CV_8UC3)
				rectangle(src, comps[i].rect.tl(), comps[i].rect.br(), Scalar( 0, 255, 0), 1, 8);
			else
				rectangle(src, comps[i].rect.tl(), comps[i].rect.br(), Scalar( 255 ), 1, 8);
		}
	}

	void groups_draw(Mat &src, vector<Rect> &groups)
	{
		for (int i=(int)groups.size()-1; i>=0; i--)
		{
			if (src.type() == CV_8UC3)
				rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 0, 255, 255 ), 3, 8 );
			else
				rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 255 ), 3, 8 );
		}
	}

	void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions)
	{
		for (int c=0; c<(int)channels.size(); c++)
		{
			Mat dst = Mat::zeros(channels[0].rows+2,channels[0].cols+2,CV_8UC1);
			for (int r=0; r<(int)regions[c].size(); r++)
			{
				ERStat er = regions[c][r];
				if (er.parent != NULL) // deprecate the root region
				{
					int newmaskval = 255;
					int flags = 4 + (newmaskval << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
					floodFill(channels[c],dst,Point(er.pixel%channels[c].cols,er.pixel/channels[c].cols),
								Scalar(255),0,Scalar(er.level),Scalar(0),flags);
				}
			}
			char buff[10]; char *buff_ptr = buff;
			sprintf(buff, "channel %d", c);
			imshow(buff_ptr, dst);
		}
		waitKey(-1);
	}

}

