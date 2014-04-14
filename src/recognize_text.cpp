#include "recognize_text.hpp"

namespace recognize_text
{

	void recognize_text(Mat &src)
	{	

		// Extract channels to be processed individually
		vector<Mat> channels;
		computeNMChannels(src, channels);

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

		//erGrouping(channels, regions, "models/trained_classifier_erGrouping.xml", 0.5, groups);

		// draw components
		Mat display_im = src.clone();
		for (int c=0; c<(int)channels.size(); c++)
		{
			src = display_im.clone();
			components_draw(src, regions[c]);
			imshow("components", src);
			waitKey();
		}

		// draw groups
		//groups_draw(src, groups);
		//imshow("grouping",src);

		er_nms(regions, 0.3);

		cout << "Done!" << endl << endl;
		cout << "Press 'e' to show the extracted Extremal Regions, any other key to exit." << endl << endl;
		if( waitKey (-1) == 101)
			er_show(channels,regions);

		// memory clean-up
		er_filter1.release();
		er_filter2.release();
		regions.clear();
		//if (!groups.empty())
		//{
		//	groups.clear();
		//}

	}

	void components_draw(Mat &src, vector<ERStat> &comps)
	{
		for (int i=0; i<(int)comps.size(); i++)
		{
			if (src.type() == CV_8UC3)
				rectangle(src, comps[i].rect.tl(), comps[i].rect.br(), Scalar( 0, 255, 0), 3, 8);
			else
				rectangle(src, comps[i].rect.tl(), comps[i].rect.br(), Scalar( 255 ), 3, 8);
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

