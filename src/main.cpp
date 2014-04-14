#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stack>
#include <set>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "perceptual_text_grouping.hpp"
#include "text_candidate_detection.hpp"

using namespace cv;

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

    clock_t t;
    t = std::clock();
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
      t = std::clock() - t;
      std::cout << "\r" << 1.0 / (((float)t) / CLOCKS_PER_SEC) << " fps     " << std::flush;
      t = std::clock();

      std::vector<Rect *> text_bounding_boxes;
      text_bounding_boxes = text_candidate_detection::text_candidate_detection(frame);
      perceptual_text_grouping::perceptual_text_grouping(frame, text_bounding_boxes);
  
      char k = waitKey(30);
      if (k == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
          break; 
      // If user closes window
      if (!cvGetWindowHandle("Test Window"))
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
}
