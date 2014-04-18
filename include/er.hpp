#ifndef __ER_H
#define __ER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <deque>

namespace er
{

	/*!
    Extremal Region Stat structure

    The ERStat structure represents a class-specific Extremal Region (ER).

    An ER is a 4-connected set of pixels with all its grey-level values smaller than the values
    in its outer boundary. A class-specific ER is selected (using a classifier) from all the ER's
    in the component tree of the image.
	*/
	struct ERStat
	{
	public:
			//! Constructor
			explicit ERStat(int level = 256, int pixel = 0, int x = 0, int y = 0);
			//! Destructor
			~ERStat() { }

			//! seed point and the threshold (max grey-level value)
			int pixel;
			int level;

			//! incrementally computable features
			int area;
			int perimeter;
			int euler;                 //!< euler number
			cv::Rect rect;
			double raw_moments[2];     //!< order 1 raw moments to derive the centroid
			double central_moments[3]; //!< order 2 central moments to construct the covariance matrix
			std::deque<int> *crossings;//!< horizontal crossings
			float med_crossings;       //!< median of the crossings at three different height levels

			//! 2nd stage features
			float hole_area_ratio;
			float convex_hull_ratio;
			float num_inflexion_points;

			// TODO Other features can be added (average color, standard deviation, and such)


			// TODO shall we include the pixel list whenever available (i.e. after 2nd stage) ?
			std::vector<int> *pixels;

			//! probability that the ER belongs to the class we are looking for
			double probability;

			//! pointers preserving the tree structure of the component tree
			ERStat* parent;
			ERStat* child;
			ERStat* next;
			ERStat* prev;

			//! wenever the regions is a local maxima of the probability
			bool local_maxima;
			ERStat* max_probability_ancestor;
			ERStat* min_probability_ancestor;
	};

	// computeNMChannels operation modes
	enum 
	{ 
		ERFILTER_NM_RGBLGrad = 0,
		ERFILTER_NM_IHSGrad  = 1
	};

	/*!
		Compute the different channels to be processed independently in the N&M algorithm
		Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

		In N&M algorithm, the combination of intensity (I), hue (H), saturation (S), and gradient
		magnitude channels (Grad) are used in order to obtain high localization recall.
		This implementation also provides an alternative combination of red (R), green (G), blue (B),
		lightness (L), and gradient magnitude (Grad).

		\param  _src           Source image. Must be RGB CV_8UC3.
		\param  _channels      Output vector<Mat> where computed channels are stored.
		\param  _mode          Mode of operation. Currently the only available options are
													 ERFILTER_NM_RGBLGrad (by default) and ERFILTER_NM_IHSGrad.
	*/
	void computeNMChannels(cv::InputArray _src, cv::OutputArrayOfArrays _channels, int _mode = ERFILTER_NM_RGBLGrad);

	// Gradient magnitude channel introduced by
  // Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
	void  get_gradient_magnitude(cv::Mat& _grey_img, cv::Mat& _gradient_magnitude);

} //end namespace er

#endif
