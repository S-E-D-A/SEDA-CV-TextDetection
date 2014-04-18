#include "er.hpp"

using namespace std;
using namespace cv;

namespace er
{
	/*!
    Extremal Region Stat structure

    The ERStat structure represents a class-specific Extremal Region (ER).

    An ER is a 4-connected set of pixels with all its grey-level values smaller than the values
    in its outer boundary. A class-specific ER is selected (using a classifier) from all the ER's
    in the component tree of the image.
	*/
	ERStat::ERStat(Mat im, int init_level, int init_pixel, int init_x, int init_y) : 
							   pixel(init_pixel),level(init_level), area(0), perimeter(0), euler(0), 
								 probability(1.0), parent(0), child(0), next(0), prev(0), local_maxima(0), 
								 max_probability_ancestor(0), min_probability_ancestor(0)
	{
			im_ptr = new Mat(im);
			rect = Rect(init_x,init_y,1,1);
			raw_moments[0] = 0.0;
			raw_moments[1] = 0.0;
			central_moments[0] = 0.0;
			central_moments[1] = 0.0;
			central_moments[2] = 0.0;
			crossings = new deque<int>();
			crossings->push_back(0);
	}

	/* ------------------------------------------------------------------------------------*/
	/* -------------------------------- Compute Channels NM -------------------------------*/
	/* ------------------------------------------------------------------------------------*/

	/*!
			Compute the diferent channels to be processed independently in the N&M algorithm
			Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

			In N&M algorithm, the combination of intensity (I), hue (H), saturation (S), and gradient
			magnitude channels (Grad) are used in order to obatin high localization recall.
			This implementation also the alternative combination of red (R), grren (G), blue (B),
			lightness (L), and gradient magnitude (Grad).

			\param  _src           Source image. Must be RGB CV_8UC3.
			\param  _channels      Output vector<Mat> where computed channels are stored.
			\param  _mode          Mode of operation. Currently the only available options are
														 ERFILTER_NM_RGBLGrad and ERFILTER_NM_IHSGrad.

	*/
	void computeNMChannels(InputArray _src, OutputArrayOfArrays _channels, int _mode)
	{

			CV_Assert( ( _mode == ERFILTER_NM_RGBLGrad ) || ( _mode == ERFILTER_NM_IHSGrad ) );

			Mat src = _src.getMat();
			if( src.empty() )
			{
					_channels.release();
					return;
			}

			// assert RGB image
			CV_Assert(src.type() == CV_8UC3);

			if (_mode == ERFILTER_NM_IHSGrad)
			{
					_channels.create( 4, 1, src.depth());

					Mat hsv;
					cvtColor(src, hsv, COLOR_RGB2HSV);
					vector<Mat> channelsHSV;
					split(hsv, channelsHSV);

					for (int i = 0; i < src.channels(); i++)
					{
							_channels.create(src.rows, src.cols, CV_8UC1, i);
							Mat channel = _channels.getMat(i);
							channelsHSV.at(i).copyTo(channel);
					}

					Mat grey;
					cvtColor(src, grey, COLOR_RGB2GRAY);
					Mat gradient_magnitude = Mat_<float>(grey.size());
					get_gradient_magnitude( grey, gradient_magnitude);
					gradient_magnitude.convertTo(gradient_magnitude, CV_8UC1);

					_channels.create(src.rows, src.cols, CV_8UC1, 3);
					Mat channelGrad = _channels.getMat(3);
					gradient_magnitude.copyTo(channelGrad);

			} else if (_mode == ERFILTER_NM_RGBLGrad) {

					_channels.create( 5, 1, src.depth());

					vector<Mat> channelsRGB;
					split(src, channelsRGB);
					for (int i = 0; i < src.channels(); i++)
					{
							_channels.create(src.rows, src.cols, CV_8UC1, i);
							Mat channel = _channels.getMat(i);
							channelsRGB.at(i).copyTo(channel);
					}

					Mat hls;
					cvtColor(src, hls, COLOR_RGB2HLS);
					vector<Mat> channelsHLS;
					split(hls, channelsHLS);

					_channels.create(src.rows, src.cols, CV_8UC1, 3);
					Mat channelL = _channels.getMat(3);
					channelsHLS.at(1).copyTo(channelL);

					Mat grey;
					cvtColor(src, grey, COLOR_RGB2GRAY);
					Mat gradient_magnitude = Mat_<float>(grey.size());
					get_gradient_magnitude( grey, gradient_magnitude);
					gradient_magnitude.convertTo(gradient_magnitude, CV_8UC1);

					_channels.create(src.rows, src.cols, CV_8UC1, 4);
					Mat channelGrad = _channels.getMat(4);
					gradient_magnitude.copyTo(channelGrad);
			}
	}


	// Gradient magnitude channel introduced by
  // Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
	void get_gradient_magnitude(Mat& _grey_img, Mat& _gradient_magnitude)
	{
			Mat C = Mat_<float>(_grey_img);

			Mat kernel = (Mat_<float>(1,3) << -1,0,1);
			Mat grad_x;
			filter2D(C, grad_x, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT);

			Mat kernel2 = (Mat_<float>(3,1) << -1,0,1);
			Mat grad_y;
			filter2D(C, grad_y, -1, kernel2, Point(-1,-1), 0, BORDER_DEFAULT);

			magnitude( grad_x, grad_y, _gradient_magnitude);
	}

} //end namespace er

