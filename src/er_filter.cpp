/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

//#include "precomp.hpp"

#include "er_filter.hpp"
#include <fstream>

#if defined _MSC_VER && _MSC_VER == 1500
    typedef int int_fast32_t;
#else
    #ifndef INT32_MAX
    #define __STDC_LIMIT_MACROS
    #include <stdint.h>
    #endif
#endif

using namespace std;

namespace cv
{

ERStat::ERStat(int init_level, int init_pixel, int init_x, int init_y) : pixel(init_pixel),
               level(init_level), area(0), perimeter(0), euler(0), probability(1.0),
               parent(0), child(0), next(0), prev(0), local_maxima(0),
               max_probability_ancestor(0), min_probability_ancestor(0)
{
    rect = Rect(init_x,init_y,1,1);
    raw_moments[0] = 0.0;
    raw_moments[1] = 0.0;
    central_moments[0] = 0.0;
    central_moments[1] = 0.0;
    central_moments[2] = 0.0;
    crossings = new std::deque<int>();
    crossings->push_back(0);
}


// derivative classes


// the classe implementing the interface for the 1st and 2nd stages of Neumann and Matas algorithm
class ERFilterNM : public ERFilter
{
public:
    //Constructor
    ERFilterNM();
    //Destructor
    ~ERFilterNM() {}

    float minProbability;
    bool  nonMaxSuppression;
    float minProbabilityDiff;

    // the key method. Takes image on input, vector of ERStat is output for the first stage,
    // input/output - for the second one.
    void run( InputArray image, std::vector<ERStat>& regions );

protected:
    int thresholdDelta;
    float maxArea;
    float minArea;

    Ptr<ERFilter::Callback> classifier;

    // count of the rejected/accepted regions
    int num_rejected_regions;
    int num_accepted_regions;

public:

    // set/get methods to set the algorithm properties,
    void setCallback(const Ptr<ERFilter::Callback>& cb);
    void setThresholdDelta(int thresholdDelta);
    void setMinArea(float minArea);
    void setMaxArea(float maxArea);
    void setMinProbability(float minProbability);
    void setMinProbabilityDiff(float minProbabilityDiff);
    void setNonMaxSuppression(bool nonMaxSuppression);
    int  getNumRejected();

private:
    // pointer to the input/output regions vector
    std::vector<ERStat> *regions;
    // image mask used for feature calculations
    Mat region_mask;

    // extract the component tree and store all the ER regions
    void er_tree_extract( InputArray image );
    // accumulate a pixel into an ER
    void er_add_pixel( ERStat *parent, int x, int y, int non_boundary_neighbours,
                       int non_boundary_neighbours_horiz,
                       int d_C1, int d_C2, int d_C3 );
    // merge an ER with its nested parent
    void er_merge( ERStat *parent, ERStat *child );
    // copy extracted regions into the output vector
    ERStat* er_save( ERStat *er, ERStat *parent, ERStat *prev );
    // recursively walk the tree and filter (remove) regions using the callback classifier
    ERStat* er_tree_filter( InputArray image, ERStat *stat, ERStat *parent, ERStat *prev );
    // recursively walk the tree selecting only regions with local maxima probability
    ERStat* er_tree_nonmax_suppression( ERStat *er, ERStat *parent, ERStat *prev );
};


// default 1st stage classifier
class ERClassifierNM1 : public ERFilter::Callback
{
public:
    //Constructor
    ERClassifierNM1(const std::string& filename);
    // Destructor
    ~ERClassifierNM1() {}

    // The classifier must return probability measure for the region.
    double eval(const ERStat& stat);

private:
    CvBoost boost;
};

// default 2nd stage classifier
class ERClassifierNM2 : public ERFilter::Callback
{
public:
    //constructor
    ERClassifierNM2(const std::string& filename);
    // Destructor
    ~ERClassifierNM2() {}

    // The classifier must return probability measure for the region.
    double eval(const ERStat& stat);

private:
    CvBoost boost;
};





// default constructor
ERFilterNM::ERFilterNM()
{
    thresholdDelta = 1;
    minArea = 0.;
    maxArea = 1.;
    minProbability = 0.;
    nonMaxSuppression = false;
    minProbabilityDiff = 1.;
    num_accepted_regions = 0;
    num_rejected_regions = 0;
}

// the key method. Takes image on input, vector of ERStat is output for the first stage,
// input/output for the second one.
void ERFilterNM::run( InputArray image, std::vector<ERStat>& _regions )
{

    // assert correct image type
    CV_Assert( image.getMat().type() == CV_8UC1 );

    regions = &_regions;
    region_mask = Mat::zeros(image.getMat().rows+2, image.getMat().cols+2, CV_8UC1);

    // if regions vector is empty we must extract the entire component tree
    if ( regions->size() == 0 )
    {
        er_tree_extract( image );
        if (nonMaxSuppression)
        {
            vector<ERStat> aux_regions;
            regions->swap(aux_regions);
            regions->reserve(aux_regions.size());
            er_tree_nonmax_suppression( &aux_regions.front(), NULL, NULL );
            aux_regions.clear();
        }
    }
    else // if regions vector is already filled we'll just filter the current regions
    {
        // the tree root must have no parent
        CV_Assert( regions->front().parent == NULL );

        vector<ERStat> aux_regions;
        regions->swap(aux_regions);
        regions->reserve(aux_regions.size());
        er_tree_filter( image, &aux_regions.front(), NULL, NULL );
        aux_regions.clear();
    }
}

// extract the component tree and store all the ER regions
// uses the algorithm described in
// Linear time maximally stable extremal regions, D Nistér, H Stewénius – ECCV 2008
void ERFilterNM::er_tree_extract( InputArray image )
{

    Mat src = image.getMat();
    // assert correct image type
    CV_Assert( src.type() == CV_8UC1 );

    if (thresholdDelta > 1)
    {
        src = (src / thresholdDelta) -1;
    }

    const unsigned char * image_data = src.data;
    int width = src.cols, height = src.rows;

    // the component stack
    vector<ERStat*> er_stack;

    //the quads for euler number calculation
    unsigned char quads[3][4];
    quads[0][0] = 1 << 3;
    quads[0][1] = 1 << 2;
    quads[0][2] = 1 << 1;
    quads[0][3] = 1;
    quads[1][0] = (1<<2)|(1<<1)|(1);
    quads[1][1] = (1<<3)|(1<<1)|(1);
    quads[1][2] = (1<<3)|(1<<2)|(1);
    quads[1][3] = (1<<3)|(1<<2)|(1<<1);
    quads[2][0] = (1<<2)|(1<<1);
    quads[2][1] = (1<<3)|(1);
    quads[2][3] = 255;


    // masks to know if a pixel is accessible and if it has been already added to some region
    vector<bool> accessible_pixel_mask(width * height);
    vector<bool> accumulated_pixel_mask(width * height);

    // heap of boundary pixels
    vector<int> boundary_pixes[256];
    vector<int> boundary_edges[256];

    // add a dummy-component before start
    er_stack.push_back(new ERStat);

    // we'll look initially for all pixels with grey-level lower than a grey-level higher than any allowed in the image
    int threshold_level = (255/thresholdDelta)+1;

    // starting from the first pixel (0,0)
    int current_pixel = 0;
    int current_edge = 0;
    int current_level = image_data[0];
    accessible_pixel_mask[0] = true;

    bool push_new_component = true;

    for (;;) {

        int x = current_pixel % width;
        int y = current_pixel / width;

        // push a component with current level in the component stack
        if (push_new_component)
            er_stack.push_back(new ERStat(current_level, current_pixel, x, y));
        push_new_component = false;

        // explore the (remaining) edges to the neighbors to the current pixel
        for ( ; current_edge < 4; current_edge++)
        {

            int neighbour_pixel = current_pixel;

						// Update neighbor_pixel according to current_edge (enum)
            switch (current_edge)
            {
                    case 0: if (x < width - 1) neighbour_pixel = current_pixel + 1;  break;			//Right
                    case 1: if (y < height - 1) neighbour_pixel = current_pixel + width; break; //Down
                    case 2: if (x > 0) neighbour_pixel = current_pixel - 1; break;							//Left
                    default: if (y > 0) neighbour_pixel = current_pixel - width; break;					//Up
            }

            // if neighbour is not accessible, mark it accessible and retreive its grey-level value
            if ( !accessible_pixel_mask[neighbour_pixel] && (neighbour_pixel != current_pixel) )
            {

                int neighbour_level = image_data[neighbour_pixel];
                accessible_pixel_mask[neighbour_pixel] = true;

                // if neighbour level is not lower than current level add neighbour to the boundary heap
                if (neighbour_level >= current_level)
                {

                    boundary_pixes[neighbour_level].push_back(neighbour_pixel);
                    boundary_edges[neighbour_level].push_back(0);

                    // if neighbour level is lower than our threshold_level set threshold_level to neighbour level
                    if (neighbour_level < threshold_level)
                        threshold_level = neighbour_level;

                }
                else // if neighbour level is lower than current add current_pixel (and next edge)
                     // to the boundary heap for later processing
                {

                    boundary_pixes[current_level].push_back(current_pixel);
                    boundary_edges[current_level].push_back(current_edge + 1);

                    // if neighbour level is lower than threshold_level set threshold_level to neighbour level
                    if (current_level < threshold_level)
                        threshold_level = current_level;

                    // consider the new pixel and its grey-level as current pixel
                    current_pixel = neighbour_pixel;
                    current_edge = 0;
                    current_level = neighbour_level;

                    // and push a new component
                    push_new_component = true;
                    break;
                }
            }

        } // else neigbor was already accessible

        if (push_new_component) continue;


        // once here we can add the current pixel to the component at the top of the stack
        // but first we find how many of its neighbours are part of the region boundary (needed for
        // perimeter and crossings calc.) and the increment in quads counts for euler number calc.
        int non_boundary_neighbours = 0;
        int non_boundary_neighbours_horiz = 0;

        unsigned char quad_before[4] = {0,0,0,0};
        unsigned char quad_after[4] = {0,0,0,0};
        quad_after[0] = 1<<1;
        quad_after[1] = 1<<3;
        quad_after[2] = 1<<2;
        quad_after[3] = 1;

        for (int edge = 0; edge < 8; edge++)
        {
            int neighbour4 = -1;
            int neighbour8 = -1;
            int cell = 0;
            switch (edge)
            {
                    case 0: if (x < width - 1) { neighbour4 = neighbour8 = current_pixel + 1;} cell = 5; break;
                    case 1: if ((x < width - 1)&&(y < height - 1)) { neighbour8 = current_pixel + 1 + width;} cell = 8; break;
                    case 2: if (y < height - 1) { neighbour4 = neighbour8 = current_pixel + width;} cell = 7; break;
                    case 3: if ((x > 0)&&(y < height - 1)) { neighbour8 = current_pixel - 1 + width;} cell = 6; break;
                    case 4: if (x > 0) { neighbour4 = neighbour8 = current_pixel - 1;} cell = 3; break;
                    case 5: if ((x > 0)&&(y > 0)) { neighbour8 = current_pixel - 1 - width;} cell = 0; break;
                    case 6: if (y > 0) { neighbour4 = neighbour8 = current_pixel - width;} cell = 1; break;
                    default: if ((x < width - 1)&&(y > 0)) { neighbour8 = current_pixel + 1 - width;} cell = 2; break;
            }
            if ((neighbour4 != -1)&&(accumulated_pixel_mask[neighbour4])&&(image_data[neighbour4]<=image_data[current_pixel]))
            {
                non_boundary_neighbours++;
                if ((edge == 0) || (edge == 4))
                    non_boundary_neighbours_horiz++;
            }

            int pix_value = image_data[current_pixel] + 1;
            if (neighbour8 != -1)
            {
                if (accumulated_pixel_mask[neighbour8])
                    pix_value = image_data[neighbour8];
            }

            if (pix_value<=image_data[current_pixel])
            {
                switch(cell)
                {
                    case 0:
                        quad_before[3] = quad_before[3] | (1<<3);
                        quad_after[3]  = quad_after[3]  | (1<<3);
                        break;
                    case 1:
                        quad_before[3] = quad_before[3] | (1<<2);
                        quad_after[3]  = quad_after[3]  | (1<<2);
                        quad_before[0] = quad_before[0] | (1<<3);
                        quad_after[0]  = quad_after[0]  | (1<<3);
                        break;
                    case 2:
                        quad_before[0] = quad_before[0] | (1<<2);
                        quad_after[0]  = quad_after[0]  | (1<<2);
                        break;
                    case 3:
                        quad_before[3] = quad_before[3] | (1<<1);
                        quad_after[3]  = quad_after[3]  | (1<<1);
                        quad_before[2] = quad_before[2] | (1<<3);
                        quad_after[2]  = quad_after[2]  | (1<<3);
                        break;
                    case 5:
                        quad_before[0] = quad_before[0] | (1);
                        quad_after[0]  = quad_after[0]  | (1);
                        quad_before[1] = quad_before[1] | (1<<2);
                        quad_after[1]  = quad_after[1]  | (1<<2);
                        break;
                    case 6:
                        quad_before[2] = quad_before[2] | (1<<1);
                        quad_after[2]  = quad_after[2]  | (1<<1);
                        break;
                    case 7:
                        quad_before[2] = quad_before[2] | (1);
                        quad_after[2]  = quad_after[2]  | (1);
                        quad_before[1] = quad_before[1] | (1<<1);
                        quad_after[1]  = quad_after[1]  | (1<<1);
                        break;
                    default:
                        quad_before[1] = quad_before[1] | (1);
                        quad_after[1]  = quad_after[1]  | (1);
                        break;
                }
            }

        }

        int C_before[3] = {0, 0, 0};
        int C_after[3] = {0, 0, 0};

        for (int p=0; p<3; p++)
        {
            for (int q=0; q<4; q++)
            {
                if ( (quad_before[0] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[1] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[2] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[3] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;

                if ( (quad_after[0] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if ( (quad_after[1] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if ( (quad_after[2] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
                if ( (quad_after[3] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_after[p]++;
            }
        }

        int d_C1 = C_after[0]-C_before[0];
        int d_C2 = C_after[1]-C_before[1];
        int d_C3 = C_after[2]-C_before[2];

        er_add_pixel(er_stack.back(), x, y, non_boundary_neighbours, non_boundary_neighbours_horiz, d_C1, d_C2, d_C3);
        accumulated_pixel_mask[current_pixel] = true;

        // if we have processed all the possible threshold levels (the hea is empty) we are done!
        if (threshold_level == (255/thresholdDelta)+1)
        {

            // save the extracted regions into the output vector
            regions->reserve(num_accepted_regions+1);
            er_save(er_stack.back(), NULL, NULL);

            // clean memory
            for (size_t r=0; r<er_stack.size(); r++)
            {
                ERStat *stat = er_stack.at(r);
                if (stat->crossings)
                {
                    stat->crossings->clear();
                    delete(stat->crossings);
                    stat->crossings = NULL;
                }
                delete stat;
            }
            er_stack.clear();

            return;
        }


        // pop the heap of boundary pixels
        current_pixel = boundary_pixes[threshold_level].back();
        boundary_pixes[threshold_level].erase(boundary_pixes[threshold_level].end()-1);
        current_edge  = boundary_edges[threshold_level].back();
        boundary_edges[threshold_level].erase(boundary_edges[threshold_level].end()-1);

        while (boundary_pixes[threshold_level].empty() && (threshold_level < (255/thresholdDelta)+1))
            threshold_level++;


        int new_level = image_data[current_pixel];

        // if the new pixel has higher grey value than the current one
        if (new_level != current_level) {

            current_level = new_level;

            // process components on the top of the stack until we reach the higher grey-level
            while (er_stack.back()->level < new_level)
            {
                ERStat* er = er_stack.back();
                er_stack.erase(er_stack.end()-1);

                if (new_level < er_stack.back()->level)
                {
                    er_stack.push_back(new ERStat(new_level, current_pixel, current_pixel%width, current_pixel/width));
                    er_merge(er_stack.back(), er);
                    break;
                }

                er_merge(er_stack.back(), er);
            }

        }

    }
}

// accumulate a pixel into an ER
void ERFilterNM::er_add_pixel(ERStat *parent, int x, int y, int non_border_neighbours,
                                                            int non_border_neighbours_horiz,
                                                            int d_C1, int d_C2, int d_C3)
{
    parent->area++;
    parent->perimeter += 4 - 2*non_border_neighbours;

    if (parent->crossings->size()>0)
    {
        if (y<parent->rect.y) parent->crossings->push_front(2);
        else if (y>parent->rect.br().y-1) parent->crossings->push_back(2);
        else {
            parent->crossings->at(y - parent->rect.y) += 2-2*non_border_neighbours_horiz;
        }
    } else {
        parent->crossings->push_back(2);
    }

    parent->euler += (d_C1 - d_C2 + 2*d_C3) / 4;

    int new_x1 = min(parent->rect.x,x);
    int new_y1 = min(parent->rect.y,y);
    int new_x2 = max(parent->rect.br().x-1,x);
    int new_y2 = max(parent->rect.br().y-1,y);
    parent->rect.x = new_x1;
    parent->rect.y = new_y1;
    parent->rect.width  = new_x2-new_x1+1;
    parent->rect.height = new_y2-new_y1+1;

    parent->raw_moments[0] += x;
    parent->raw_moments[1] += y;

    parent->central_moments[0] += x * x;
    parent->central_moments[1] += x * y;
    parent->central_moments[2] += y * y;
}

// merge an ER with its nested parent
void ERFilterNM::er_merge(ERStat *parent, ERStat *child)
{

    parent->area += child->area;

    parent->perimeter += child->perimeter;


    for (int i=parent->rect.y; i<=min(parent->rect.br().y-1,child->rect.br().y-1); i++)
        if (i-child->rect.y >= 0)
            parent->crossings->at(i-parent->rect.y) += child->crossings->at(i-child->rect.y);

    for (int i=parent->rect.y-1; i>=child->rect.y; i--)
        if (i-child->rect.y < (int)child->crossings->size())
            parent->crossings->push_front(child->crossings->at(i-child->rect.y));
        else
            parent->crossings->push_front(0);

    for (int i=parent->rect.br().y; i<child->rect.y; i++)
        parent->crossings->push_back(0);

    for (int i=max(parent->rect.br().y,child->rect.y); i<=child->rect.br().y-1; i++)
        parent->crossings->push_back(child->crossings->at(i-child->rect.y));

    parent->euler += child->euler;

    int new_x1 = min(parent->rect.x,child->rect.x);
    int new_y1 = min(parent->rect.y,child->rect.y);
    int new_x2 = max(parent->rect.br().x-1,child->rect.br().x-1);
    int new_y2 = max(parent->rect.br().y-1,child->rect.br().y-1);
    parent->rect.x = new_x1;
    parent->rect.y = new_y1;
    parent->rect.width  = new_x2-new_x1+1;
    parent->rect.height = new_y2-new_y1+1;

    parent->raw_moments[0] += child->raw_moments[0];
    parent->raw_moments[1] += child->raw_moments[1];

    parent->central_moments[0] += child->central_moments[0];
    parent->central_moments[1] += child->central_moments[1];
    parent->central_moments[2] += child->central_moments[2];

    vector<int> m_crossings;
    m_crossings.push_back(child->crossings->at((int)(child->rect.height)/6));
    m_crossings.push_back(child->crossings->at((int)3*(child->rect.height)/6));
    m_crossings.push_back(child->crossings->at((int)5*(child->rect.height)/6));
    std::sort(m_crossings.begin(), m_crossings.end());
    child->med_crossings = (float)m_crossings.at(1);

    // free unnecessary mem
    child->crossings->clear();
    delete(child->crossings);
    child->crossings = NULL;

    // recover the original grey-level
    child->level = child->level*thresholdDelta;

    // before saving calculate P(child|character) and filter if possible
    if (classifier != NULL)
    {
        child->probability = classifier->eval(*child);
    }

    if ( (((classifier!=NULL)?(child->probability >= minProbability):true)||(nonMaxSuppression)) &&
         ((child->area >= (minArea*region_mask.rows*region_mask.cols)) &&
          (child->area <= (maxArea*region_mask.rows*region_mask.cols)) &&
          (child->rect.width > 2) && (child->rect.height > 2)) )
    {

        num_accepted_regions++;

        child->next = parent->child;
        if (parent->child)
            parent->child->prev = child;
        parent->child = child;
        child->parent = parent;

    } else {

        num_rejected_regions++;

        if (child->prev !=NULL)
            child->prev->next = child->next;

        ERStat *new_child = child->child;
        if (new_child != NULL)
        {
            while (new_child->next != NULL)
                new_child = new_child->next;
            new_child->next = parent->child;
            if (parent->child)
                parent->child->prev = new_child;
            parent->child   = child->child;
            child->child->parent = parent;
        }

        // free mem
        if(child->crossings)
        {
            child->crossings->clear();
            delete(child->crossings);
            child->crossings = NULL;
        }
        delete(child);
    }

}

// copy extracted regions into the output vector
ERStat* ERFilterNM::er_save( ERStat *er, ERStat *parent, ERStat *prev )
{

    regions->push_back(*er);

    regions->back().parent = parent;
    if (prev != NULL)
    {
      prev->next = &(regions->back());
    }
    else if (parent != NULL)
      parent->child = &(regions->back());

    ERStat *old_prev = NULL;
    ERStat *this_er  = &regions->back();

    if (this_er->parent == NULL)
    {
       this_er->probability = 0;
    }

    if (nonMaxSuppression)
    {
        if (this_er->parent == NULL)
        {
            this_er->max_probability_ancestor = this_er;
            this_er->min_probability_ancestor = this_er;
        }
        else
        {
            this_er->max_probability_ancestor = (this_er->probability > parent->max_probability_ancestor->probability)? this_er :  parent->max_probability_ancestor;

            this_er->min_probability_ancestor = (this_er->probability < parent->min_probability_ancestor->probability)? this_er :  parent->min_probability_ancestor;

            if ( (this_er->max_probability_ancestor->probability > minProbability) && (this_er->max_probability_ancestor->probability - this_er->min_probability_ancestor->probability > minProbabilityDiff))
            {
              this_er->max_probability_ancestor->local_maxima = true;
              if ((this_er->max_probability_ancestor == this_er) && (this_er->parent->local_maxima))
              {
                this_er->parent->local_maxima = false;
              }
            }
            else if (this_er->probability < this_er->parent->probability)
            {
              this_er->min_probability_ancestor = this_er;
            }
            else if (this_er->probability > this_er->parent->probability)
            {
              this_er->max_probability_ancestor = this_er;
            }


        }
    }

    for (ERStat * child = er->child; child; child = child->next)
    {
        old_prev = er_save(child, this_er, old_prev);
    }

    return this_er;
}

// recursively walk the tree and filter (remove) regions using the callback classifier
ERStat* ERFilterNM::er_tree_filter ( InputArray image, ERStat * stat, ERStat *parent, ERStat *prev )
{
    Mat src = image.getMat();
    // assert correct image type
    CV_Assert( src.type() == CV_8UC1 );

    //Fill the region and calculate 2nd stage features
    Mat region = region_mask(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x+2,stat->rect.br().y+2)));
    region = Scalar(0);
    int newMaskVal = 255;
    int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
    Rect rect;

	Mat m = Mat(src(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x,stat->rect.br().y))));
    floodFill( m,
               region, Point(stat->pixel%src.cols - stat->rect.x, stat->pixel/src.cols - stat->rect.y),
               Scalar(255), &rect, Scalar(stat->level), Scalar(0), flags );
    rect.width += 2;
    rect.height += 2;
    region = region(rect);

    vector<vector<Point> > contours;
    vector<Point> contour_poly;
    vector<Vec4i> hierarchy;
    findContours( region, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0) );
    //TODO check epsilon parameter of approxPolyDP (set empirically) : we want more precission
    //     if the region is very small because otherwise we'll loose all the convexities
    approxPolyDP( Mat(contours[0]), contour_poly, (float)min(rect.width,rect.height)/17, true );

    bool was_convex = false;
    int  num_inflexion_points = 0;

    for (int p = 0 ; p<(int)contour_poly.size(); p++)
    {
        int p_prev = p-1;
        int p_next = p+1;
        if (p_prev == -1)
            p_prev = (int)contour_poly.size()-1;
        if (p_next == (int)contour_poly.size())
            p_next = 0;

        double angle_next = atan2((double)(contour_poly[p_next].y-contour_poly[p].y),
                                  (double)(contour_poly[p_next].x-contour_poly[p].x));
        double angle_prev = atan2((double)(contour_poly[p_prev].y-contour_poly[p].y),
                                  (double)(contour_poly[p_prev].x-contour_poly[p].x));
        if ( angle_next < 0 )
            angle_next = 2.*CV_PI + angle_next;

        double angle = (angle_next - angle_prev);
        if (angle > 2.*CV_PI)
            angle = angle - 2.*CV_PI;
        else if (angle < 0)
            angle = 2.*CV_PI + std::abs(angle);

        if (p>0)
        {
            if ( ((angle > CV_PI)&&(!was_convex)) || ((angle < CV_PI)&&(was_convex)) )
                num_inflexion_points++;
        }
        was_convex = (angle > CV_PI);

    }

    floodFill(region, Point(0,0), Scalar(255), 0);
    int holes_area = region.cols*region.rows-countNonZero(region);

    int hull_area = 0;

    {

        vector<Point> hull;
        convexHull(contours[0], hull, false);
        hull_area = (int)contourArea(hull);
    }


    stat->hole_area_ratio = (float)holes_area / stat->area;
    stat->convex_hull_ratio = (float)hull_area / (float)contourArea(contours[0]);
    stat->num_inflexion_points = (float)num_inflexion_points;


    // calculate P(child|character) and filter if possible
    if ( (classifier != NULL) && (stat->parent != NULL) )
    {
        stat->probability = classifier->eval(*stat);
    }

    if ( ( ((classifier != NULL)?(stat->probability >= minProbability):true) &&
          ((stat->area >= minArea*region_mask.rows*region_mask.cols) &&
           (stat->area <= maxArea*region_mask.rows*region_mask.cols)) ) ||
        (stat->parent == NULL) )
    {

        num_accepted_regions++;
        regions->push_back(*stat);

        regions->back().parent = parent;
        regions->back().next   = NULL;
        regions->back().child  = NULL;

        if (prev != NULL)
            prev->next = &(regions->back());
        else if (parent != NULL)
            parent->child = &(regions->back());

        ERStat *old_prev = NULL;
        ERStat *this_er  = &regions->back();

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_filter(image, child, this_er, old_prev);
        }

        return this_er;

    } else {

        num_rejected_regions++;

        ERStat *old_prev = prev;

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_filter(image, child, parent, old_prev);
        }

        return old_prev;
    }

}

// recursively walk the tree selecting only regions with local maxima probability
ERStat* ERFilterNM::er_tree_nonmax_suppression ( ERStat * stat, ERStat *parent, ERStat *prev )
{

    if ( ( stat->local_maxima ) || ( stat->parent == NULL ) )
    {

        regions->push_back(*stat);

        regions->back().parent = parent;
        regions->back().next   = NULL;
        regions->back().child  = NULL;

        if (prev != NULL)
            prev->next = &(regions->back());
        else if (parent != NULL)
            parent->child = &(regions->back());

        ERStat *old_prev = NULL;
        ERStat *this_er  = &regions->back();

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression( child, this_er, old_prev );
        }

        return this_er;

    } else {

        num_rejected_regions++;
        num_accepted_regions--;

        ERStat *old_prev = prev;

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression( child, parent, old_prev );
        }

        return old_prev;
    }

}

void ERFilterNM::setCallback(const Ptr<ERFilter::Callback>& cb)
{
    classifier = cb;
}

void ERFilterNM::setMinArea(float _minArea)
{
    CV_Assert( (_minArea >= 0) && (_minArea < maxArea) );
    minArea = _minArea;
    return;
}

void ERFilterNM::setMaxArea(float _maxArea)
{
    CV_Assert(_maxArea <= 1);
    CV_Assert(minArea < _maxArea);
    maxArea = _maxArea;
    return;
}

void ERFilterNM::setThresholdDelta(int _thresholdDelta)
{
    CV_Assert( (_thresholdDelta > 0) && (_thresholdDelta <= 128) );
    thresholdDelta = _thresholdDelta;
    return;
}

void ERFilterNM::setMinProbability(float _minProbability)
{
    CV_Assert( (_minProbability >= 0.0) && (_minProbability <= 1.0) );
    minProbability = _minProbability;
    return;
}

void ERFilterNM::setMinProbabilityDiff(float _minProbabilityDiff)
{
    CV_Assert( (_minProbabilityDiff >= 0.0) && (_minProbabilityDiff <= 1.0) );
    minProbabilityDiff = _minProbabilityDiff;
    return;
}

void ERFilterNM::setNonMaxSuppression(bool _nonMaxSuppression)
{
    nonMaxSuppression = _nonMaxSuppression;
    return;
}

int ERFilterNM::getNumRejected()
{
    return num_rejected_regions;
}




// load default 1st stage classifier if found
ERClassifierNM1::ERClassifierNM1(const std::string& filename)
{

    if (ifstream(filename.c_str()))
        boost.load( filename.c_str(), "boost" );
    else
        CV_Error(CV_StsBadArg, "Default classifier file not found!");
}

double ERClassifierNM1::eval(const ERStat& stat)
{
    //Classify
    float arr[] = {0,(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings};

    vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );

    float votes = boost.predict( Mat(sample), Mat(), Range::all(), false, true );

    // Logistic Correction returns a probability value (in the range(0,1))
    return (double)1-(double)1/(1+exp(-2*votes));
}


// load default 2nd stage classifier if found
ERClassifierNM2::ERClassifierNM2(const std::string& filename)
{
    if (ifstream(filename.c_str()))
        boost.load( filename.c_str(), "boost" );
    else
        CV_Error(CV_StsBadArg, "Default classifier file not found!");
}

double ERClassifierNM2::eval(const ERStat& stat)
{
    //Classify
    float arr[] = {0,(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings, stat.hole_area_ratio,
                     stat.convex_hull_ratio, stat.num_inflexion_points};

    vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );

    float votes = boost.predict( Mat(sample), Mat(), Range::all(), false, true );

    // Logistic Correction returns a probability value (in the range(0,1))
    return (double)1-(double)1/(1+exp(-2*votes));
}


/*!
    Create an Extremal Region Filter for the 1st stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    The component tree of the image is extracted by a threshold increased step by step
    from 0 to 255, incrementally computable descriptors (aspect_ratio, compactness,
    number of holes, and number of horizontal crossings) are computed for each ER
    and used as features for a classifier which estimates the class-conditional
    probability P(er|character). The value of P(er|character) is tracked using the inclusion
    relation of ER across all thresholds and only the ERs which correspond to local maximum
    of the probability P(er|character) are selected (if the local maximum of the
    probability is above a global limit pmin and the difference between local maximum and
    local minimum is greater than minProbabilityDiff).

    \param  cb                Callback with the classifier.
                              default classifier can be implicitly load with function loadClassifierNM1()
                              from file in samples/cpp/trained_classifierNM1.xml
    \param  thresholdDelta    Threshold step in subsequent thresholds when extracting the component tree
    \param  minArea           The minimum area (% of image size) allowed for retreived ER's
    \param  minArea           The maximum area (% of image size) allowed for retreived ER's
    \param  minProbability    The minimum probability P(er|character) allowed for retreived ER's
    \param  nonMaxSuppression Whenever non-maximum suppression is done over the branch probabilities
    \param  minProbability    The minimum probability difference between local maxima and local minima ERs
*/
Ptr<ERFilter> createERFilterNM1(const Ptr<ERFilter::Callback>& cb, int thresholdDelta,
                                float minArea, float maxArea, float minProbability,
                                bool nonMaxSuppression, float minProbabilityDiff)
{

    CV_Assert( (minProbability >= 0.) && (minProbability <= 1.) );
    CV_Assert( (minArea < maxArea) && (minArea >=0.) && (maxArea <= 1.) );
    CV_Assert( (thresholdDelta >= 0) && (thresholdDelta <= 128) );
    CV_Assert( (minProbabilityDiff >= 0.) && (minProbabilityDiff <= 1.) );

	Ptr<ERFilter> filter = new ERFilterNM();

    filter->setCallback(cb);

    filter->setThresholdDelta(thresholdDelta);
    filter->setMinArea(minArea);
    filter->setMaxArea(maxArea);
    filter->setMinProbability(minProbability);
    filter->setNonMaxSuppression(nonMaxSuppression);
    filter->setMinProbabilityDiff(minProbabilityDiff);
    return filter;
}

/*!
    Create an Extremal Region Filter for the 2nd stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    In the second stage, the ERs that passed the first stage are classified into character
    and non-character classes using more informative but also more computationally expensive
    features. The classifier uses all the features calculated in the first stage and the following
    additional features: hole area ratio, convex hull ratio, and number of outer inflexion points.

    \param  cb             Callback with the classifier
                           default classifier can be implicitly load with function loadClassifierNM1()
                           from file in samples/cpp/trained_classifierNM2.xml
    \param  minProbability The minimum probability P(er|character) allowed for retreived ER's
*/
Ptr<ERFilter> createERFilterNM2(const Ptr<ERFilter::Callback>& cb, float minProbability)
{

    CV_Assert( (minProbability >= 0.) && (minProbability <= 1.) );

	Ptr<ERFilter> filter = new ERFilterNM();

    filter->setCallback(cb);

    filter->setMinProbability(minProbability);
    return filter;
}

/*!
    Allow to implicitly load the default classifier when creating an ERFilter object.
    The function takes as parameter the XML or YAML file with the classifier model
    (e.g. trained_classifierNM1.xml) returns a pointer to ERFilter::Callback.
*/
Ptr<ERFilter::Callback> loadClassifierNM1(const std::string& filename)

{
	return Ptr<ERFilter::Callback>(new ERClassifierNM1(filename));
}

/*!
    Allow to implicitly load the default classifier when creating an ERFilter object.
    The function takes as parameter the XML or YAML file with the classifier model
    (e.g. trained_classifierNM2.xml) returns a pointer to ERFilter::Callback.
*/
Ptr<ERFilter::Callback> loadClassifierNM2(const std::string& filename)
{
	return Ptr<ERFilter::Callback>(new ERClassifierNM2(filename));
}


/* ------------------------------------------------------------------------------------*/
/* -------------------------------- Compute Channels NM -------------------------------*/
/* ------------------------------------------------------------------------------------*/



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



// --------------------------- Word line filtering ---------------------------------

#define PI 3.14159265

const double DIST_MAX_RATIO = 2;
const double DIST_MIN_RATIO = 0.2;
const double HEIGHT_RATIO = 2;
const double HORIZ_ANGLE = 20;
const double MS_DELAY = 50;
const int MIN_WORD_LENGTH = 4;

struct ERChar
{
	ERStat stat;
	int channel;
	int ID;
};

struct ERWord
{
	vector<ERChar> letters;
};

//bool sortByChannel(ERChar erc1, ERChar erc2)
//{
//	return (erc1.channel > erc2.channel);
//}
struct ptrstat_x_cmp 
{
	bool operator() (const Ptr<ERChar> erc1, const Ptr<ERChar> erc2)
	{
		return (erc1->stat.rect.tl().x < erc2->stat.rect.tl().x);
	}
};

bool sortByX(Ptr<ERChar> erc1, Ptr<ERChar> erc2)
{
		return (erc1->stat.rect.tl().x < erc2->stat.rect.tl().x);
}

typedef set<Ptr<ERChar>, ptrstat_x_cmp> ERChar_set;

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

bool compareERChar_sets(ERChar_set s1, ERChar_set s2)
{
	ERChar_set::iterator it1, it2;

	it1 = s1.begin();
	it2 = s2.begin();
	while ( it1 != s1.end() && it2 != s2.end() )
	{
		//const ERChar* er1_ptr = (*it1)Ptr<ERChar>();
		//const ERChar* er1_ptr = (*it1)(ERChar*());
		//const ERChar* er2_ptr = (*it2)Ptr<ERChar>();
		//if ( er1_ptr != er2_ptr)
		if ( (*it1) != (*it2) )
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

void erShow(int rows, int cols, vector<Mat> &channels, ERChar_set &er_set, double delay)
{
	vector<Mat> masks;
	for (int c=0; c<(int)channels.size(); c++)
	{
		Mat blank = Mat::zeros(channels[c].rows+2,channels[0].cols+2,CV_8UC1);
		masks.push_back(blank);
	}

	//vector<Point> pts;
	//Point leftmost, rightmost;
	//leftmost.x = cols;
	//rightmost.x = 0;
	ERChar_set::iterator it;
	for (it = er_set.begin(); it != er_set.end(); it++)
	{
		ERStat er = (*it)->stat;
		int c = (*it)->channel;
		if (er.parent != NULL)
		{
			int newmaskval = 255;
			int flags = 4 + (newmaskval << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
			floodFill(channels[c],masks[c],Point(er.pixel%channels[c].cols,
						er.pixel/channels[c].cols), Scalar(255),0,Scalar(er.level),Scalar(0),flags);
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


	Mat out = Mat::zeros(rows, cols, CV_8UC3);
	vector<Mat> clr_channels;
	split(out, clr_channels);
	for (int c=0; c<(int)channels.size(); c++)
	{
		Mat disp(masks[c], Range(1, masks[c].rows-1), Range(1, masks[c].cols-1));
		if (c==0)
			clr_channels[0] = clr_channels[0] + disp;
		else
			clr_channels[1] = clr_channels[1] + disp;
		merge(clr_channels, out);
	}

	imshow("Regions", out);
	cvMoveWindow("Regions", 200, 50);
	waitKey(delay);

}	

bool v1(Ptr<ERChar> er1, Ptr<ERChar> er2)
{
	Rect& r1 = er1->stat.rect;
	Rect& r2 = er2->stat.rect;

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
bool isSubWord(ERChar_set s1, ERChar_set s2)
{
	if (s1.empty() || s2.empty())
		return false;

	ERChar_set::iterator it1 = s1.begin();
	ERChar_set::iterator it2 = s2.begin();
	
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

void pruneSubwords(vector<list<ERChar_set> > &words)
{

	for (int d=0; d<(int)words.size(); d++)
	{

		// Clear minimum word length words
		if ( d < MIN_WORD_LENGTH-2 )
		{
			words[d].clear();
			continue;
		}

		list<ERChar_set>::iterator it1;
		list<ERChar_set>::iterator it2;

		bool erased = true;
		for (it1 = words[d-1].begin(); it1 != words[d-1].end(); )
		{

			ERChar_set small = (*it1);
			for(it2 = words[d].begin(); it2 != words[d].end(); it2++)
			{
				ERChar_set big = (*it2);
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

void erWordLine(Mat &img, vector<Mat> &channels, vector<vector<ERStat> > &regions)
{

	CV_Assert( !img.empty() );
	CV_Assert( !regions.empty() );

	int ROWS = img.rows;
	int COLS = img.cols;

	ERChar_set er_all;
	int count = 0;
	for (int i=0; i<(int)regions.size(); i++)
	{
		for (int j=0; j<(int)regions[i].size(); j++)
		{
			Ptr<ERChar> erc = new ERChar();
			erc->stat = regions[i][j];
			erc->channel = i;
			erc->ID = count;
			er_all.insert(erc);
			count++;
		}
	}

	// Show all regions
	erShow(ROWS, COLS, channels, er_all, 0);

	vector<list<ERChar_set> > words;
	// Vector of list of sets to store candidate words
	// Outer vector:	determines length
	// 								[0] all length 1 words or subwords
	// 								[1] all length 2 words or subwords
	// 																:
	//																 
	// Inner list:	words or subwords of that length

	list<ERChar_set> words_of_length;

	// Create pairwise words
	{
		ERChar_set::iterator it1, it2;
		it1 = er_all.begin();
		while ( it1 != er_all.end() ) 
		{
			if ((*it1)->stat.parent == NULL)
			{
				cout << "it1 null parent " << endl;
				it1++;
			}
			it2 = it1;
			it2++;

			ERChar_set pair;
			while ( it2 != er_all.end() )
			{
				Ptr<ERChar> er1 = *it1;
				Ptr<ERChar> er2 = *it2;
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
		list<ERChar_set>::iterator it1, it2;
		it1	= words[d-1].begin();	
		for (it1 = words[d-1].begin(); it1 != words[d-1].end(); it1++ )
		{
			// Word or subword at it1
			ERChar_set er_set1 = *it1;
				
			// First letter of word or subword at it1
			Ptr<ERChar> subset_1;
			subset_1 = (*er_set1.begin());

			// Generate the subword from n=2,...,N at it1
			ERChar_set subset_2N;
			ERChar_set::iterator it_mid_1;
			it_mid_1 = er_set1.begin();
			it_mid_1++;
			for ( ; it_mid_1 != er_set1.end(); it_mid_1++)	
			{
				subset_2N.insert(*it_mid_1);
			}

			// New word or subword to be added	
			ERChar_set subset_1N;
			for (it2 = it1; it2 != words[d-1].end(); it2++ )
			{
				// Word or subword at it2
				ERChar_set er_set2 = *it2;

				// Generate subword from n=1,...,N-1 at it2
				ERChar_set subset_1N1;
				ERChar_set::iterator it_mid_2, it_end_2;
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
				Ptr<ERChar> subset_N;
				ERChar_set::iterator it_last_2 = er_set2.end();
				it_last_2--;
				subset_N = (*it_last_2);

				if ( compareERChar_sets(subset_2N, subset_1N1) )
				{
					// Insert the first letter from it1
					subset_1N.insert(subset_1);
					
					// Insert all the letters from the overlap of it1 and it2 (length N-1)
					ERChar_set::iterator it_mid;
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
	//			erShow(ROWS, COLS, channels, words[d][s], MS_DELAY);

	// Prune subwords from words
	pruneSubwords(words);

	// Show all ERs
	for (int d=0; d<(int)words.size(); d++)
	{
		list<ERChar_set>::iterator s;
		for (s=words[d].begin(); s != words[d].end(); s++)
			erShow(ROWS, COLS, channels, (*s), 0);
	}
	


}

} //namespace cv








