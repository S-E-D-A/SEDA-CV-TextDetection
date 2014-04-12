#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stack>
#include <set>
#include <vector>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "perceptual_text_grouping.hpp"
#include "text_candidate_detection.hpp"
#include "erfilter.hpp"

using namespace cv;
using namespace std;

void groups_draw(Mat &src, vector<Rect> &groups);
void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions);

int main(int argc, char ** argv) {
  /* Handles input otpions */
  /* ##################### */
  extern char *optarg;
  int c, err = 0; 

  // Whether or not -i flag set to value
  bool image_file = false;

  // Stores webcam number (-1 default is any cam)
  int webcam_number = -1;
  std::stringstream ss;

  // Store the -i flag input value
  char *image_file_name=NULL;

  // Usage string of this program
  static char usage[] = "Usage: %s -i path/to/image\n"
                        "\t\t-v webcam number (optional)\n";

  while ((c = getopt(argc, argv, "v::hi:")) != -1)
    switch (c) {
    case 'i':
      image_file = true;
      // Store the value of -i flag in variable
      image_file_name = optarg;
      break;
    case 'h':
      // Show help info and exit
      fprintf(stderr, usage, argv[0]);
      exit(1);
    case 'v':
      if (!optarg) 
        break;
      ss << optarg;
      if (!(ss >> webcam_number) || webcam_number < 0) {
        fprintf(stderr,"Webcam Number not valid. Exiting ...\n\n");
        fprintf(stderr, usage, argv[0]);
        exit(-1);
      }
      break;
    case '?':
      // Given incorrect option flag error
      err = 1;
      break;
    }

  if (err) {
    fprintf(stderr, usage, argv[0]);
    exit(1);
  }
  /* ##################### */


  if (!image_file) { 
    VideoCapture cap(webcam_number); // open any video camera

    if (!cap.isOpened())  // if not success, exit program
    {
      std::cout << "Cannot open the video cam" << std::endl;
      return -1;
    }

    namedWindow("Test Window",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

    while (1)
    {
        Mat frame;
        bool bSuccess = cap.read(frame); // read a new frame from video
      if (!bSuccess) //if not success, break loop
      {
        std::cout << "Cannot read a frame from video stream" << std::endl;
        break;
      }

      imshow("Test Window", frame); //show the frame in "MyVideo" window

      std::vector<Rect *> text_bounding_boxes;
      text_bounding_boxes = text_candidate_detection::text_candidate_detection(frame);
      perceptual_text_grouping::perceptual_text_grouping(frame, text_bounding_boxes);

      if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
          break; 
    }

    return 0;
  }    

  Mat src = imread(image_file_name, 1);
  namedWindow("Test Window", CV_WINDOW_AUTOSIZE);
  clock_t t;
  t = std::clock();
  std::vector<Rect *> text_bounding_boxes;
  text_bounding_boxes = text_candidate_detection::text_candidate_detection(src);
  
  perceptual_text_grouping::perceptual_text_grouping(src, text_bounding_boxes);
  t = std::clock() - t;
  printf ("It took me %d clicks (%f seconds).\n",(int)t,((float)t)/CLOCKS_PER_SEC);
  
  // Wait for escape keypress
  while (true) {

    int c;
    c = waitKey(20);
    if( (char)c == 27 )
      { break; }
  
    // If user closes window
    if (!cvGetWindowHandle("Test Window"))
      break;
  } 




	// --------- EXTREMAL REGION FILTERING -----------
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

    // Detect character groups
    cout << "Grouping extracted ERs ... ";
    vector<Rect> groups;
    erGrouping(channels, regions, "models/trained_classifier_erGrouping.xml", 0.5, groups);

    // draw groups
    groups_draw(src, groups);
    imshow("grouping",src);

    cout << "Done!" << endl << endl;
    cout << "Press 'e' to show the extracted Extremal Regions, any other key to exit." << endl << endl;
    if( waitKey (-1) == 101)
        er_show(channels,regions);

    // memory clean-up
    er_filter1.release();
    er_filter2.release();
    regions.clear();
    if (!groups.empty())
    {
        groups.clear();
    }

	return 0;
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
                int newMaskVal = 255;
                int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
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
