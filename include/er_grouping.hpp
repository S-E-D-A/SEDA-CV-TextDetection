#ifndef __ER_GROUPING_H
#define __ER_GROUPING_H

#include "er_filter.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <fstream>


/*!
    Find groups of Extremal Regions that are organized as text blocks. This function implements
    the grouping algorithm described in:
    Gomez L. and Karatzas D.: Multi-script Text Extraction from Natural Scenes, ICDAR 2013.
    Notice that this implementation constrains the results to horizontally-aligned text and
    latin script (since ERFilter classifiers are trained only for latin script detection).

    The algorithm combines two different clustering techniques in a single parameter-free procedure
    to detect groups of regions organized as text. The maximally meaningful groups are fist detected
    in several feature spaces, where each feature space is a combination of proximity information
    (x,y coordinates) and a similarity measure (intensity, color, size, gradient magnitude, etc.),
    thus providing a set of hypotheses of text groups. Evidence Accumulation framework is used to
    combine all these hypotheses to get the final estimate. Each of the resulting groups are finally
    validated using a classifier in order to assest if they form a valid horizontally-aligned text block.

    \param  src            Vector of sinle channel images CV_8UC1 from wich the regions were extracted.
    \param  regions        Vector of ER's retreived from the ERFilter algorithm from each channel
    \param  filename       The XML or YAML file with the classifier model (e.g. trained_classifier_erGrouping.xml)
    \param  minProbability The minimum probability for accepting a group
    \param  groups         The output of the algorithm are stored in this parameter as list of rectangles.
*/


void erGrouping(cv::InputArrayOfArrays src, std::vector<std::vector<er::ERStat> > &regions,
                const std::string& filename, float minProbablity, std::vector<cv::Rect> &groups);

#endif
