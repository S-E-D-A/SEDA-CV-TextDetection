#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <stack>
#include <set>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "text_candidate_detection.hpp"

const int ADAPT_THRESHOLD_WINDOW_SIZE = 17;
const double SOBEL_THRESHOLD = 40.0;
const double TEXTURE_THRESHOLD = 0.5;
const double MIN_AREA_THRESHOLD = 600.0;
const double FRACTION_OF_IMAGE_THRESHOLD = 0.5;

using namespace cv;

namespace text_candidate_detection {

  std::vector<Rect *> text_candidate_detection(Mat src) {

    // Convert to grayscale
    Mat src_gray;
    cvtColor(src, src_gray, CV_RGB2GRAY);

    // Apply thresholding
    Mat thresh_im;
    adaptiveThreshold(src_gray, thresh_im, 255, ADAPTIVE_THRESH_MEAN_C, 
                        THRESH_BINARY, ADAPT_THRESHOLD_WINDOW_SIZE, 0);

    // Get connected components in tree hierarchy
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(thresh_im, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // Generate Sobel Edges 
    Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(src_gray, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);

    // Gradient Magnitude of Sobel Edges 
    Mat sobel_magnitude;
    Mat sobel_angle;
    cartToPolar(grad_x,grad_y,sobel_magnitude,sobel_angle);

    // Stack used for traversing the contour tree
    std::stack<int> dfs_contour;

    // Set holds candidate text regions
    std::set<int> text_candidate_contours;

    // Find first contour in first hierarchy
    for (size_t i = 0; i < contours.size(); i++) {
      if (hierarchy[i][1] == -1) {
        dfs_contour.push(i);
        //std::cout << "Pushed " << i << std::endl;
        break;
      }
    }

    // Traversal
    while (!dfs_contour.empty()) {

      // Pop current contour
      int curr_contour = dfs_contour.top();
      dfs_contour.pop();

      /*
      *  Region filtering
      */ 

      // Size
      Rect contour_bb = boundingRect(contours[curr_contour]);
      if (!size_filter(contour_bb, src)) {

      } else if (border_energy(contours[curr_contour], sobel_magnitude) < SOBEL_THRESHOLD) {
        //std::cout << "rejecting region: border energy too low" << std::endl;
      } else if (eigen_texture_measure(contours[curr_contour], src_gray, 8, 8) < TEXTURE_THRESHOLD) {

      } else {

        // Add to candidates
        text_candidate_contours.insert(curr_contour);

        // Remove parents
        int parent_contour = hierarchy[curr_contour][3];
        while (parent_contour != -1) {
          text_candidate_contours.erase(parent_contour);
          parent_contour = hierarchy[parent_contour][3];
        }
      }

      // Push neighbor
      if (hierarchy[curr_contour][0] != -1) {
        dfs_contour.push(hierarchy[curr_contour][0]);
      }

      // Push Children (This gets traversed first, if possible)
      if (hierarchy[curr_contour][2] != -1) {
        int child_contour = hierarchy[hierarchy[curr_contour][2]][2];
        if (child_contour != -1)
          dfs_contour.push(child_contour);
        else 
          dfs_contour.push(hierarchy[curr_contour][2]);
      }
    }

    // Output a list of bounding boxes of candidates
    std::vector<Rect*> text_bbs;  
    for (auto it = text_candidate_contours.begin(); it != text_candidate_contours.end(); ++it) {
      Rect * r = new Rect;
      *r = boundingRect(contours[*it]);
      text_bbs.push_back(r);
    }

    return text_bbs;
  }

  double border_energy(vector<Point> & points, Mat & sobel_image) {
    double border_energy = 0.0;
    for (auto itr = points.begin(); itr != points.end(); ++itr) {
      border_energy += (float)sobel_image.at<float>(*itr);
    }

    border_energy = border_energy / (double)points.size();

    //std::cout << "BE: " << border_energy << std::endl;

    return border_energy;
  }

  // Computes the Eigen-Transform texture operator for the image block im
  // l : number of eigenvalues to disgard
  double eigen_texture_measure(vector<Point> & points, Mat im, int n, int l) {

    Rect contour_bb = boundingRect(points);
    int w = round(contour_bb.size().height / 4.0);

    double texture_measure = 0;
    for (int i = 0; i < n; ++i) {

      int idx = rand() % points.size(); 

      Point sample = points[idx];

      if ((sample.x - w < 0) || (sample.x + w >= im.size().width)) {
        --n;
        continue;
      }

      if ((sample.y - w < 0) || (sample.y + w >= im.size().height)) {
        --n;
        continue;
      }
       
      Rect roi(sample.x - w, sample.y - w, w, w); 

      Mat image_roi = im(roi);
      Mat image_roi_float;
      image_roi.convertTo(image_roi_float, CV_32F);

      std::vector<double> singular_vals; 

      // We only need the singular values
      SVD::compute(image_roi_float, singular_vals, SVD::NO_UV); 

      int w = singular_vals.size();

      double eigen_val_sum = 0;
      for (int j = l; j < w ; ++j) {
        eigen_val_sum += singular_vals[j];
      }
      texture_measure += eigen_val_sum / (1 + w - l);
    }

    //std::cout << "ET: " << texture_measure << std::endl;
    if (n == 0)
      return 0;
    texture_measure /= n;
    return texture_measure;
  }

  bool size_filter(Rect contour_bb, const Mat& src) {

    if (contour_bb.size().area() > FRACTION_OF_IMAGE_THRESHOLD * src.rows * src.cols) {
      return false;
    } else if (contour_bb.size().area() < MIN_AREA_THRESHOLD) {
      return false;
      //std::cout << "rejecting region: too small" << std::endl;
    }

    return true;
  }
}
