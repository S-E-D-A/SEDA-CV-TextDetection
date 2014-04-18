#include "er_grouping.hpp"

/*  NFA approximation functions */

// ln(10)
#ifndef M_LN10
#define M_LN10     2.30258509299404568401799145468436421
#endif
// Doubles relative error factor
#define RELATIVE_ERROR_FACTOR 100.0

using namespace cv;
using namespace std;

// Compare doubles by relative error.
static int double_equal(double a, double b)
{
    double abs_diff,aa,bb,abs_max;

    /* trivial case */
    if( a == b ) return true;

    abs_diff = fabs(a-b);
    aa = fabs(a);
    bb = fabs(b);
    abs_max = aa > bb ? aa : bb;

    /* DBL_MIN is the smallest normalized number, thus, the smallest
       number whose relative error is bounded by DBL_EPSILON. For
       smaller numbers, the same quantization steps as for DBL_MIN
       are used. Then, for smaller numbers, a meaningful "relative"
       error should be computed by dividing the difference by DBL_MIN. */
    if( abs_max < DBL_MIN ) abs_max = DBL_MIN;

    /* equal if relative error <= factor x eps */
    return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}


/*
     Computes the natural logarithm of the absolute value of
     the gamma function of x using the Lanczos approximation.
     See http://www.rskey.org/gamma.htm
*/
static double log_gamma_lanczos(double x)
{
    static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                           8687.24529705, 1168.92649479, 83.8676043424,
                           2.50662827511 };
    double a = (x+0.5) * log(x+5.5) - (x+5.5);
    double b = 0.0;
    int n;

    for(n=0;n<7;n++)
    {
        a -= log( x + (double) n );
        b += q[n] * pow( x, (double) n );
    }
    return a + log(b);
}

/*
     Computes the natural logarithm of the absolute value of
     the gamma function of x using Windschitl method.
     See http://www.rskey.org/gamma.htm
*/
static double log_gamma_windschitl(double x)
{
    return 0.918938533204673 + (x-0.5)*log(x) - x
           + 0.5*x*log( x*sinh(1/x) + 1/(810.0*pow(x,6.0)) );
}

/*
     Computes the natural logarithm of the absolute value of
     the gamma function of x. When x>15 use log_gamma_windschitl(),
     otherwise use log_gamma_lanczos().
*/
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

// Size of the table to store already computed inverse values.
#define TABSIZE 100000

/*
     Computes -log10(NFA).
     NFA stands for Number of False Alarms:
*/
static double NFA(int n, int k, double p, double logNT)
{
    static double inv[TABSIZE];   /* table to keep computed inverse values */
    double tolerance = 0.1;       /* an error of 10% in the result is accepted */
    double log1term,term,bin_term,mult_term,bin_tail,err,p_term;
    int i;

    if (p<=0)
        p=0.000000000000000000000000000001;
    if (p>=1)
        p=0.999999999999999999999999999999;

    /* check parameters */
    if( n<0 || k<0 || k>n || p<=0.0 || p>=1.0 )
    {
        CV_Error(CV_StsBadArg, "erGrouping wrong n, k or p values in NFA call!");
    }

    /* trivial cases */
    if( n==0 || k==0 ) return -logNT;
    if( n==k ) return -logNT - (double) n * log10(p);

    /* probability term */
    p_term = p / (1.0-p);

    /* compute the first term of the series */
    log1term = log_gamma( (double) n + 1.0 ) - log_gamma( (double) k + 1.0 )
               - log_gamma( (double) (n-k) + 1.0 )
               + (double) k * log(p) + (double) (n-k) * log(1.0-p);
    term = exp(log1term);

    /* in some cases no more computations are needed */
    if( double_equal(term,0.0) )              /* the first term is almost zero */
    {
        if( (double) k > (double) n * p )     /* at begin or end of the tail?  */
            return -log1term / M_LN10 - logNT;  /* end: use just the first term  */
        else
            return -logNT;                      /* begin: the tail is roughly 1  */
    }

    /* compute more terms if needed */
    bin_tail = term;
    for(i=k+1;i<=n;i++)
    {
        bin_term = (double) (n-i+1) * ( i<TABSIZE ?
                    ( inv[i]!=0.0 ? inv[i] : ( inv[i] = 1.0 / (double) i ) ) :
                    1.0 / (double) i );

        mult_term = bin_term * p_term;
        term *= mult_term;
        bin_tail += term;
        if(bin_term<1.0)
        {
            err = term * ( ( 1.0 - pow( mult_term, (double) (n-i+1) ) ) /
                           (1.0-mult_term) - 1.0 );
            if( err < tolerance * fabs(-log10(bin_tail)-logNT) * bin_tail ) break;
        }
    }
    return -log10(bin_tail) - logNT;
}


// Minibox : smallest enclosing box of a set of n points in d dimensions

class Minibox {
private:
    vector<float> edge_begin;
    vector<float> edge_end;
    bool   initialized;

public:
    // creates an empty box
    Minibox();

    // copies p to the internal point set
    void check_in (vector<float> *p);

    // returns the volume of the box
    long double volume();
};

Minibox::Minibox()
{
    initialized = false;
}

void Minibox::check_in (vector<float> *p)
{
    if(!initialized) for (int i=0; i<(int)p->size(); i++)
    {
        edge_begin.push_back(p->at(i));
        edge_end.push_back(p->at(i)+0.00000000000000001f);
        initialized = true;
    }
    else for (int i=0; i<(int)p->size(); i++)
    {
        edge_begin.at(i) = min(p->at(i),edge_begin.at(i));
        edge_end.at(i) = max(p->at(i),edge_end.at(i));
    }
}

long double Minibox::volume ()
{
    long double volume_ = 1;
    for (int i=0; i<(int)edge_begin.size(); i++)
    {
        volume_ = volume_ * (edge_end.at(i) - edge_begin.at(i));
    }
    return (volume_);
}


#define MAX_NUM_FEATURES 9


/*  Hierarchical Clustering classes and functions */


// Hierarchical Clustering linkage variants
enum method_codes
{
    METHOD_METR_SINGLE           = 0,
    METHOD_METR_AVERAGE          = 1
};

#ifndef INT32_MAX
#define MAX_INDEX 0x7fffffffL
#else
#define MAX_INDEX INT32_MAX
#endif

// A node in the hierarchical clustering algorithm
struct node {
    int_fast32_t node1, node2;
    double dist;

    inline friend bool operator< (const node a, const node b)
    {
        // Numbers are always smaller than NaNs.
        return a.dist < b.dist || (a.dist==a.dist && b.dist!=b.dist);
    }
};

// self-destructing array pointer
template <typename type>
class auto_array_ptr {
private:
    type * ptr;
public:
    auto_array_ptr() { ptr = NULL; }
    template <typename index>
    auto_array_ptr(index const size) { init(size); }
    template <typename index, typename value>
    auto_array_ptr(index const size, value const val) { init(size, val); }

    ~auto_array_ptr()
    {
        delete [] ptr;
    }
    void free() {
        delete [] ptr;
        ptr = NULL;
    }
    template <typename index>
    void init(index const size)
    {
        ptr = new type [size];
    }
    template <typename index, typename value>
    void init(index const size, value const val)
    {
        init(size);
        for (index i=0; i<size; i++) ptr[i] = val;
    }
    inline operator type *() const { return ptr; }
};

// The result of the hierarchical clustering algorithm
class cluster_result {
private:
    auto_array_ptr<node> Z;
    int_fast32_t pos;

public:
    cluster_result(const int_fast32_t size): Z(size)
    {
        pos = 0;
    }

    void append(const int_fast32_t node1, const int_fast32_t node2, const double dist)
    {
        Z[pos].node1 = node1;
        Z[pos].node2 = node2;
        Z[pos].dist  = dist;
        pos++;
    }

    node * operator[] (const int_fast32_t idx) const { return Z + idx; }

    void sqrt() const
    {
        for (int_fast32_t i=0; i<pos; i++)
            Z[i].dist = ::sqrt(Z[i].dist);
    }

    void sqrt(const double) const  // ignore the argument
    {
        sqrt();
    }
};

// Class for a doubly linked list
class doubly_linked_list {
public:
    int_fast32_t start;
    auto_array_ptr<int_fast32_t> succ;

private:
    auto_array_ptr<int_fast32_t> pred;

public:
    doubly_linked_list(const int_fast32_t size): succ(size+1), pred(size+1)
    {
        for (int_fast32_t i=0; i<size; i++)
        {
            pred[i+1] = i;
            succ[i] = i+1;
        }
        start = 0;
    }

    void remove(const int_fast32_t idx)
    {
        // Remove an index from the list.
        if (idx==start)
        {
            start = succ[idx];
        } else {
            succ[pred[idx]] = succ[idx];
            pred[succ[idx]] = pred[idx];
        }
        succ[idx] = 0; // Mark as inactive
    }

    bool is_inactive(int_fast32_t idx) const
    {
        return (succ[idx]==0);
    }
};

// Indexing functions
// D is the upper triangular part of a symmetric (NxN)-matrix
// We require r_ < c_ !
#define D_(r_,c_) ( D[(static_cast<std::ptrdiff_t>(2*N-3-(r_))*(r_)>>1)+(c_)-1] )
// Z is an ((N-1)x4)-array
#define Z_(_r, _c) (Z[(_r)*4 + (_c)])

/*
   Lookup function for a union-find data structure.

   The function finds the root of idx by going iteratively through all
   parent elements until a root is found. An element i is a root if
   nodes[i] is zero. To make subsequent searches faster, the entry for
   idx and all its parents is updated with the root element.
*/
class union_find {
private:
    auto_array_ptr<int_fast32_t> parent;
    int_fast32_t nextparent;

public:
    void init(const int_fast32_t size)
    {
        parent.init(2*size-1, 0);
        nextparent = size;
    }

    int_fast32_t Find (int_fast32_t idx) const
    {
        if (parent[idx] !=0 ) // a -> b
        {
            int_fast32_t p = idx;
            idx = parent[idx];
            if (parent[idx] !=0 ) // a -> b -> c
            {
                do
                {
                    idx = parent[idx];
                } while (parent[idx] != 0);
                do
                {
                    int_fast32_t tmp = parent[p];
                    parent[p] = idx;
                    p = tmp;
                } while (parent[p] != idx);
            }
        }
        return idx;
    }

    void Union (const int_fast32_t node1, const int_fast32_t node2)
    {
        parent[node1] = parent[node2] = nextparent++;
    }
};


/* Functions for the update of the dissimilarity array */

inline static void f_single( double * const b, const double a )
{
    if (*b > a) *b = a;
}
inline static void f_average( double * const b, const double a, const double s, const double t)
{
    *b = s*a + t*(*b);
}

/*
     This is the NN-chain algorithm.

     N: integer
     D: condensed distance matrix N*(N-1)/2
     Z2: output data structure
*/
template <const unsigned char method, typename t_members>
static void NN_chain_core(const int_fast32_t N, double * const D, t_members * const members, cluster_result & Z2)
{
    int_fast32_t i;

    auto_array_ptr<int_fast32_t> NN_chain(N);
    int_fast32_t NN_chain_tip = 0;

    int_fast32_t idx1, idx2;

    double size1, size2;
    doubly_linked_list active_nodes(N);

    double min;

    for (int_fast32_t j=0; j<N-1; j++)
    {
        if (NN_chain_tip <= 3)
        {
            NN_chain[0] = idx1 = active_nodes.start;
            NN_chain_tip = 1;

            idx2 = active_nodes.succ[idx1];
            min = D_(idx1,idx2);

            for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
            {
                if (D_(idx1,i) < min)
                {
                    min = D_(idx1,i);
                    idx2 = i;
                }
            }
        }  // a: idx1   b: idx2
        else {
            NN_chain_tip -= 3;
            idx1 = NN_chain[NN_chain_tip-1];
            idx2 = NN_chain[NN_chain_tip];
            min = idx1<idx2 ? D_(idx1,idx2) : D_(idx2,idx1);
        }  // a: idx1   b: idx2

        do {
            NN_chain[NN_chain_tip] = idx2;

            for (i=active_nodes.start; i<idx2; i=active_nodes.succ[i])
            {
                if (D_(i,idx2) < min)
                {
                    min = D_(i,idx2);
                    idx1 = i;
                }
            }
            for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
            {
                if (D_(idx2,i) < min)
                {
                    min = D_(idx2,i);
                    idx1 = i;
                }
            }

            idx2 = idx1;
            idx1 = NN_chain[NN_chain_tip++];

        } while (idx2 != NN_chain[NN_chain_tip-2]);

        Z2.append(idx1, idx2, min);

        if (idx1>idx2)
        {
            int_fast32_t tmp = idx1;
            idx1 = idx2;
            idx2 = tmp;
        }

        //if ( method == METHOD_METR_AVERAGE )
        {
            size1 = static_cast<double>(members[idx1]);
            size2 = static_cast<double>(members[idx2]);
            members[idx2] += members[idx1];
        }

        // Remove the smaller index from the valid indices (active_nodes).
        active_nodes.remove(idx1);

        switch (method) {
            case METHOD_METR_SINGLE:
                /*
                 Single linkage.
                */
                // Update the distance matrix in the range [start, idx1).
                for (i=active_nodes.start; i<idx1; i=active_nodes.succ[i])
                    f_single(&D_(i, idx2), D_(i, idx1) );
                // Update the distance matrix in the range (idx1, idx2).
                for (; i<idx2; i=active_nodes.succ[i])
                    f_single(&D_(i, idx2), D_(idx1, i) );
                // Update the distance matrix in the range (idx2, N).
                for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
                    f_single(&D_(idx2, i), D_(idx1, i) );
                break;

            case METHOD_METR_AVERAGE:
            {
                /*
                Average linkage.
                */
                // Update the distance matrix in the range [start, idx1).
                double s = size1/(size1+size2);
                double t = size2/(size1+size2);
                for (i=active_nodes.start; i<idx1; i=active_nodes.succ[i])
                    f_average(&D_(i, idx2), D_(i, idx1), s, t );
                // Update the distance matrix in the range (idx1, idx2).
                for (; i<idx2; i=active_nodes.succ[i])
                    f_average(&D_(i, idx2), D_(idx1, i), s, t );
                // Update the distance matrix in the range (idx2, N).
                for (i=active_nodes.succ[idx2]; i<N; i=active_nodes.succ[i])
                    f_average(&D_(idx2, i), D_(idx1, i), s, t );
                break;
            }
        }
    }
}


/*
   Clustering methods for vector data
*/

template <typename t_dissimilarity>
static void MST_linkage_core_vector(const int_fast32_t N,
                                    t_dissimilarity & dist,
                                    cluster_result & Z2) {
/*
     Hierarchical clustering using the minimum spanning tree

     N: integer, number of data points
     dist: function pointer to the metric
     Z2: output data structure
*/
    int_fast32_t i;
    int_fast32_t idx2;
    doubly_linked_list active_nodes(N);
    auto_array_ptr<double> d(N);

    int_fast32_t prev_node;
    double min;

    // first iteration
    idx2 = 1;
    min = d[1] = dist(0,1);
    for (i=2; min!=min && i<N; i++) // eliminate NaNs if possible
    {
        min = d[i] = dist(0,i);
        idx2 = i;
    }

    for ( ; i<N; i++)
    {
        d[i] = dist(0,i);
        if (d[i] < min)
        {
            min = d[i];
            idx2 = i;
        }
    }

    Z2.append(0, idx2, min);

    for (int_fast32_t j=1; j<N-1; j++)
    {
        prev_node = idx2;
        active_nodes.remove(prev_node);

        idx2 = active_nodes.succ[0];
        min = d[idx2];

        for (i=idx2; min!=min && i<N; i=active_nodes.succ[i]) // eliminate NaNs if possible
        {
            min = d[i] = dist(i, prev_node);
            idx2 = i;
        }

        for ( ; i<N; i=active_nodes.succ[i])
        {
            double tmp = dist(i, prev_node);
            if (d[i] > tmp)
                d[i] = tmp;
            if (d[i] < min)
            {
                min = d[i];
                idx2 = i;
            }
        }
        Z2.append(prev_node, idx2, min);
    }
}

class linkage_output {
private:
    double * Z;
    int_fast32_t pos;

public:
    linkage_output(double * const _Z)
    {
         this->Z = _Z;
         pos = 0;
    }

    void append(const int_fast32_t node1, const int_fast32_t node2, const double dist, const double size)
    {
         if (node1<node2)
         {
                Z[pos++] = static_cast<double>(node1);
                Z[pos++] = static_cast<double>(node2);
         } else {
                Z[pos++] = static_cast<double>(node2);
                Z[pos++] = static_cast<double>(node1);
         }
         Z[pos++] = dist;
         Z[pos++] = size;
    }
};


/*
    Generate the specific output format for a dendrogram from the
    clustering output.

    The list of merging steps can be sorted or unsorted.
*/

// The size of a node is either 1 (a single point) or is looked up from
// one of the clusters.
#define size_(r_) ( ((r_<N) ? 1 : Z_(r_-N,3)) )

static void generate_dendrogram(double * const Z, cluster_result & Z2, const int_fast32_t N)
{
    // The array "nodes" is a union-find data structure for the cluster
    // identites (only needed for unsorted cluster_result input).
    union_find nodes;
    std::stable_sort(Z2[0], Z2[N-1]);
    nodes.init(N);

    linkage_output output(Z);
    int_fast32_t node1, node2;

    for (int_fast32_t i=0; i<N-1; i++) {
         // Get two data points whose clusters are merged in step i.
         // Find the cluster identifiers for these points.
         node1 = nodes.Find(Z2[i]->node1);
         node2 = nodes.Find(Z2[i]->node2);
         // Merge the nodes in the union-find data structure by making them
         // children of a new node.
         nodes.Union(node1, node2);
         output.append(node1, node2, Z2[i]->dist, size_(node1)+size_(node2));
    }
}

/*
     Clustering on vector data
*/

enum {
    // metrics
    METRIC_EUCLIDEAN       =  0,
    METRIC_CITYBLOCK       =  1,
    METRIC_SEUCLIDEAN      =  2,
    METRIC_SQEUCLIDEAN     =  3
};

/*
    This class handles all the information about the dissimilarity
    computation.
*/
class dissimilarity {
private:
    double * Xa;
    auto_array_ptr<double> Xnew;
    std::ptrdiff_t dim; // size_t saves many statis_cast<> in products
    int_fast32_t N;
    int_fast32_t * members;
    void (cluster_result::*postprocessfn) (const double) const;
    double postprocessarg;

    double (dissimilarity::*distfn) (const int_fast32_t, const int_fast32_t) const;

    auto_array_ptr<double> precomputed;

    double * V;
    const double * V_data;

public:
    dissimilarity (double * const _Xa, int _Num, int _dim,
                   int_fast32_t * const _members,
                   const unsigned char method,
                   const unsigned char metric,
                   bool temp_point_array)
                   : Xa(_Xa),
                     dim(_dim),
                     N(_Num),
                     members(_members),
                     postprocessfn(NULL),
                     V(NULL)
    {
        switch (method) {
            case METHOD_METR_SINGLE: // only single linkage allowed here but others may come...
            default:
                postprocessfn = NULL; // default
                switch (metric)
                {
                    case METRIC_EUCLIDEAN:
                        set_euclidean();
                        break;
                    case METRIC_SEUCLIDEAN:
                    case METRIC_SQEUCLIDEAN:
                        distfn = &dissimilarity::sqeuclidean;
                        break;
                    case METRIC_CITYBLOCK:
                        set_cityblock();
                        break;
                }
        }

        if (temp_point_array)
        {
            Xnew.init((N-1)*dim);
        }
    }

    ~dissimilarity()
    {
        free(V);
    }

    inline double operator () (const int_fast32_t i, const int_fast32_t j) const
    {
        return (this->*distfn)(i,j);
    }

    inline double X (const int_fast32_t i, const int_fast32_t j) const
    {
        return Xa[i*dim+j];
    }

    inline bool Xb (const int_fast32_t i, const int_fast32_t j) const
    {
        return  reinterpret_cast<bool *>(Xa)[i*dim+j];
    }

    inline double * Xptr(const int_fast32_t i, const int_fast32_t j) const
    {
        return Xa+i*dim+j;
    }

    void postprocess(cluster_result & Z2) const
    {
        if (postprocessfn!=NULL)
        {
            (Z2.*postprocessfn)(postprocessarg);
        }
    }

    double sqeuclidean(const int_fast32_t i, const int_fast32_t j) const
    {
        double sum = 0;
        double const * Pi = Xa+i*dim;
        double const * Pj = Xa+j*dim;
        for (int_fast32_t k=0; k<dim; k++)
        {
            double diff = Pi[k] - Pj[k];
            sum += diff*diff;
        }
        return sum;
    }

private:

    void set_euclidean()
    {
        distfn = &dissimilarity::sqeuclidean;
        postprocessfn = &cluster_result::sqrt;
    }

    void set_cityblock()
    {
        distfn = &dissimilarity::cityblock;
    }

    double seuclidean(const int_fast32_t i, const int_fast32_t j) const
    {
        double sum = 0;
        for (int_fast32_t k=0; k<dim; k++)
        {
            double diff = X(i,k)-X(j,k);
            sum += diff*diff/V_data[k];
        }
        return sum;
    }

    double cityblock(const int_fast32_t i, const int_fast32_t j) const
    {
        double sum = 0;
        for (int_fast32_t k=0; k<dim; k++)
        {
            sum += fabs(X(i,k)-X(j,k));
        }
        return sum;
    }
};

/*Clustering for the "stored matrix approach": the input is the array of pairwise dissimilarities*/
static int linkage(double *D, int N, double * Z)
{
    CV_Assert(N >=1);
    CV_Assert(N <= MAX_INDEX/4);

    try
    {

        cluster_result Z2(N-1);
        auto_array_ptr<int_fast32_t> members;
        // The distance update formula needs the number of data points in a cluster.
        members.init(N, 1);
        NN_chain_core<METHOD_METR_AVERAGE, int_fast32_t>(N, D, members, Z2);
        generate_dendrogram(Z, Z2, N);

    } // try
    catch (const std::bad_alloc&)
    {
        CV_Error(CV_StsNoMem, "Not enough Memory for erGrouping hierarchical clustering structures!");
    }
    catch(const std::exception&)
    {
        CV_Error(CV_StsError, "Uncaught exception in erGrouping!");
    }
    catch(...)
    {
        CV_Error(CV_StsError, "C++ exception (unknown reason) in erGrouping!");
    }
    return 0;

}

/*Clustering for the "stored data approach": the input are points in a vector space.*/
static int linkage_vector(double *X, int N, int dim, double * Z, unsigned char method, unsigned char metric)
{

    CV_Assert(N >=1);
    CV_Assert(N <= MAX_INDEX/4);
    CV_Assert(dim >=1);

    try
    {
        cluster_result Z2(N-1);
        auto_array_ptr<int_fast32_t> members;
        dissimilarity dist(X, N, dim, members, method, metric, false);
        MST_linkage_core_vector(N, dist, Z2);
        dist.postprocess(Z2);
        generate_dendrogram(Z, Z2, N);
    } // try
    catch (const std::bad_alloc&)
    {
        CV_Error(CV_StsNoMem, "Not enough Memory for erGrouping hierarchical clustering structures!");
    }
    catch(const std::exception&)
    {
        CV_Error(CV_StsError, "Uncaught exception in erGrouping!");
    }
    catch(...)
    {
        CV_Error(CV_StsError, "C++ exception (unknown reason) in erGrouping!");
    }
    return 0;
}


/*  Maximal Meaningful Clusters Detection */

struct HCluster{
    int num_elem;           // number of elements
    vector<int> elements;   // elements (contour ID)
    int nfa;                // the number of false alarms for this merge
    float dist;             // distance of the merge
    float dist_ext;         // distamce where this merge will merge with another
    long double volume;     // volume of the bounding sphere (or bounding box)
    long double volume_ext; // volume of the sphere(or box) + envolvent empty space
    vector<vector<float> > points; // nD points in this cluster
    bool max_meaningful;    // is this merge max meaningul ?
    vector<int> max_in_branch; // otherwise which merges are the max_meaningful in this branch
    int min_nfa_in_branch;  // min nfa detected within the chilhood
    int node1;
    int node2;
};

class MaxMeaningfulClustering
{
public:
    unsigned char method_;
    unsigned char metric_;

    /// Constructor.
    MaxMeaningfulClustering(unsigned char method, unsigned char metric){ method_=method; metric_=metric; }

    void operator()(double *data, unsigned int num, int dim, unsigned char method,
                    unsigned char metric, vector< vector<int> > *meaningful_clusters);
    void operator()(double *data, unsigned int num, unsigned char method,
                    vector< vector<int> > *meaningful_clusters);

private:
    /// Helper functions
    void build_merge_info(double *dendogram, double *data, int num, int dim, bool use_full_merge_rule,
                          vector<HCluster> *merge_info, vector< vector<int> > *meaningful_clusters);
    void build_merge_info(double *dendogram, int num, vector<HCluster> *merge_info,
                          vector< vector<int> > *meaningful_clusters);

    /// Number of False Alarms
    int nfa(float sigma, int k, int N);

};

void MaxMeaningfulClustering::operator()(double *data, unsigned int num, int dim, unsigned char method,
                                         unsigned char metric, vector< vector<int> > *meaningful_clusters)
{

    double *Z = (double*)malloc(((num-1)*4) * sizeof(double)); // we need 4 floats foreach sample merge.
    if (Z == NULL)
        CV_Error(CV_StsNoMem, "Not enough Memory for erGrouping hierarchical clustering structures!");

    linkage_vector(data, (int)num, dim, Z, method, metric);

    vector<HCluster> merge_info;
    build_merge_info(Z, data, (int)num, dim, false, &merge_info, meaningful_clusters);

    free(Z);
    merge_info.clear();
}

void MaxMeaningfulClustering::operator()(double *data, unsigned int num, unsigned char method,
                                         vector< vector<int> > *meaningful_clusters)
{

    CV_Assert(method == METHOD_METR_AVERAGE);

    double *Z = (double*)malloc(((num-1)*4) * sizeof(double)); // we need 4 floats foreach sample merge.
    if (Z == NULL)
        CV_Error(CV_StsNoMem, "Not enough Memory for erGrouping hierarchical clustering structures!");

    linkage(data, (int)num, Z);

    vector<HCluster> merge_info;
    build_merge_info(Z, (int)num, &merge_info, meaningful_clusters);

    free(Z);
    merge_info.clear();
}

void MaxMeaningfulClustering::build_merge_info(double *Z, double *X, int N, int dim,
                                               bool use_full_merge_rule,
                                               vector<HCluster> *merge_info,
                                               vector< vector<int> > *meaningful_clusters)
{

    // walk the whole dendogram
    for (int i=0; i<(N-1)*4; i=i+4)
    {
        HCluster cluster;
        cluster.num_elem = (int)Z[i+3]; //number of elements

        int node1  = (int)Z[i];
        int node2  = (int)Z[i+1];
        float dist = (float)Z[i+2];

        if (node1<N)
        {
            vector<float> point;
            for (int n=0; n<dim; n++)
                point.push_back((float)X[node1*dim+n]);
            cluster.points.push_back(point);
            cluster.elements.push_back((int)node1);
        }
        else
        {
            for (int ii=0; ii<(int)merge_info->at(node1-N).points.size(); ii++)
            {
                cluster.points.push_back(merge_info->at(node1-N).points[ii]);
                cluster.elements.push_back(merge_info->at(node1-N).elements[ii]);
            }
            //update the extended volume of node1 using the dist where this cluster merge with another
            merge_info->at(node1-N).dist_ext = dist;
        }
        if (node2<N)
        {
            vector<float> point;
            for (int n=0; n<dim; n++)
                point.push_back((float)X[node2*dim+n]);
            cluster.points.push_back(point);
            cluster.elements.push_back((int)node2);
        }
        else
        {
            for (int ii=0; ii<(int)merge_info->at(node2-N).points.size(); ii++)
            {
                cluster.points.push_back(merge_info->at(node2-N).points[ii]);
                cluster.elements.push_back(merge_info->at(node2-N).elements[ii]);
            }

            //update the extended volume of node2 using the dist where this cluster merge with another
            merge_info->at(node2-N).dist_ext = dist;
        }

        Minibox mb;
        for (int ii=0; ii<(int)cluster.points.size(); ii++)
        {
            mb.check_in(&cluster.points.at(ii));
        }

        cluster.dist   = dist;
        cluster.volume = mb.volume();
        if (cluster.volume >= 1)
            cluster.volume = 0.999999;
        if (cluster.volume == 0)
            cluster.volume = 0.001;

        cluster.volume_ext=1;

        if (node1>=N)
        {
            merge_info->at(node1-N).volume_ext = cluster.volume;
        }
        if (node2>=N)
        {
            merge_info->at(node2-N).volume_ext = cluster.volume;
        }

        cluster.node1 = node1;
        cluster.node2 = node2;

        merge_info->push_back(cluster);

    }

    for (int i=0; i<(int)merge_info->size(); i++)
    {

        merge_info->at(i).nfa = nfa((float)merge_info->at(i).volume,
                                    merge_info->at(i).num_elem, N);
        int node1 = merge_info->at(i).node1;
        int node2 = merge_info->at(i).node2;

        if ((node1<N)&&(node2<N))
        {
            // both nodes are individual samples (nfa=1) : each cluster is max.
            merge_info->at(i).max_meaningful = true;
            merge_info->at(i).max_in_branch.push_back(i);
            merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
        } else {
            if ((node1>=N)&&(node2>=N))
            {
                // both nodes are "sets" : we must evaluate the merging condition
                if ( ( (use_full_merge_rule) &&
                       ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + merge_info->at(node2-N).nfa) &&
                       (merge_info->at(i).nfa < min(merge_info->at(node1-N).min_nfa_in_branch,
                                                    merge_info->at(node2-N).min_nfa_in_branch))) ) ||
                     ( (!use_full_merge_rule) &&
                       ((merge_info->at(i).nfa < min(merge_info->at(node1-N).min_nfa_in_branch,
                                                     merge_info->at(node2-N).min_nfa_in_branch))) ) )
                {
                    merge_info->at(i).max_meaningful = true;
                    merge_info->at(i).max_in_branch.push_back(i);
                    merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                    for (int k =0; k<(int)merge_info->at(node1-N).max_in_branch.size(); k++)
                        merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;
                    for (int k =0; k<(int)merge_info->at(node2-N).max_in_branch.size(); k++)
                        merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
                } else {
                    merge_info->at(i).max_meaningful = false;
                    merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                    merge_info->at(node1-N).max_in_branch.begin(),
                    merge_info->at(node1-N).max_in_branch.end());
                    merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                    merge_info->at(node2-N).max_in_branch.begin(),
                    merge_info->at(node2-N).max_in_branch.end());

                    if (merge_info->at(i).nfa < min(merge_info->at(node1-N).min_nfa_in_branch,
                                                    merge_info->at(node2-N).min_nfa_in_branch))

                        merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                    else
                        merge_info->at(i).min_nfa_in_branch = min(merge_info->at(node1-N).min_nfa_in_branch,
                                                                  merge_info->at(node2-N).min_nfa_in_branch);
                }
            } else {

                //one of the nodes is a "set" and the other is an individual sample : check merging condition
                if (node1>=N)
                {
                    if ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + 1) &&
                        (merge_info->at(i).nfa<merge_info->at(node1-N).min_nfa_in_branch))
                    {
                        merge_info->at(i).max_meaningful = true;
                        merge_info->at(i).max_in_branch.push_back(i);
                        merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                        for (int k =0; k<(int)merge_info->at(node1-N).max_in_branch.size(); k++)
                            merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;
                    } else {
                        merge_info->at(i).max_meaningful = false;
                        merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                                                               merge_info->at(node1-N).max_in_branch.begin(),
                                                               merge_info->at(node1-N).max_in_branch.end());
                        merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,
                                                                  merge_info->at(node1-N).min_nfa_in_branch);
                    }
                } else {
                    if ((merge_info->at(i).nfa < merge_info->at(node2-N).nfa + 1) &&
                        (merge_info->at(i).nfa<merge_info->at(node2-N).min_nfa_in_branch))
                    {
                        merge_info->at(i).max_meaningful = true;
                        merge_info->at(i).max_in_branch.push_back(i);
                        merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                        for (int k =0; k<(int)merge_info->at(node2-N).max_in_branch.size(); k++)
                            merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
                    } else {
                        merge_info->at(i).max_meaningful = false;
                        merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                        merge_info->at(node2-N).max_in_branch.begin(),
                        merge_info->at(node2-N).max_in_branch.end());
                        merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,
                        merge_info->at(node2-N).min_nfa_in_branch);
                    }
                }
            }
        }
    }

    for (int i=0; i<(int)merge_info->size(); i++)
    {
        if (merge_info->at(i).max_meaningful)
        {
            vector<int> cluster;
            for (int k=0; k<(int)merge_info->at(i).elements.size();k++)
                cluster.push_back(merge_info->at(i).elements.at(k));
            meaningful_clusters->push_back(cluster);
        }
    }

}

void MaxMeaningfulClustering::build_merge_info(double *Z, int N, vector<HCluster> *merge_info,
                                               vector< vector<int> > *meaningful_clusters)
{

    // walk the whole dendogram
    for (int i=0; i<(N-1)*4; i=i+4)
    {
        HCluster cluster;
        cluster.num_elem = (int)Z[i+3]; //number of elements

        int node1  = (int)Z[i];
        int node2  = (int)Z[i+1];
        float dist = (float)Z[i+2];
        if (dist != dist) //this is to avoid NaN values
            dist=0;

        if (node1<N)
        {
            cluster.elements.push_back((int)node1);
        }
        else
        {
            for (int ii=0; ii<(int)merge_info->at(node1-N).elements.size(); ii++)
            {
                cluster.elements.push_back(merge_info->at(node1-N).elements[ii]);
            }
        }
        if (node2<N)
        {
            cluster.elements.push_back((int)node2);
        }
        else
        {
            for (int ii=0; ii<(int)merge_info->at(node2-N).elements.size(); ii++)
            {
                cluster.elements.push_back(merge_info->at(node2-N).elements[ii]);
            }
        }

        cluster.dist   = dist;
        if (cluster.dist >= 1)
            cluster.dist = 0.999999f;
        if (cluster.dist == 0)
            cluster.dist = 1.e-25f;

        cluster.dist_ext   = 1;

        if (node1>=N)
        {
            merge_info->at(node1-N).dist_ext = cluster.dist;
        }
        if (node2>=N)
        {
            merge_info->at(node2-N).dist_ext = cluster.dist;
        }

        cluster.node1 = node1;
        cluster.node2 = node2;

        merge_info->push_back(cluster);
    }

    for (int i=0; i<(int)merge_info->size(); i++)
    {

        merge_info->at(i).nfa = nfa(merge_info->at(i).dist,
                                    merge_info->at(i).num_elem, N);
        int node1 = merge_info->at(i).node1;
        int node2 = merge_info->at(i).node2;

        if ((node1<N)&&(node2<N))
        {
            // both nodes are individual samples (nfa=1) so this cluster is max.
            merge_info->at(i).max_meaningful = true;
            merge_info->at(i).max_in_branch.push_back(i);
            merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
        } else {
            if ((node1>=N)&&(node2>=N))
            {
                // both nodes are "sets" so we must evaluate the merging condition
                if ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + merge_info->at(node2-N).nfa) &&
                    (merge_info->at(i).nfa < min(merge_info->at(node1-N).min_nfa_in_branch,
                                                 merge_info->at(node2-N).min_nfa_in_branch)))
                {
                    merge_info->at(i).max_meaningful = true;
                    merge_info->at(i).max_in_branch.push_back(i);
                    merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                    for (int k =0; k<(int)merge_info->at(node1-N).max_in_branch.size(); k++)
                        merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;
                    for (int k =0; k<(int)merge_info->at(node2-N).max_in_branch.size(); k++)
                        merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
                } else {
                    merge_info->at(i).max_meaningful = false;
                    merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                    merge_info->at(node1-N).max_in_branch.begin(),
                    merge_info->at(node1-N).max_in_branch.end());
                    merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                    merge_info->at(node2-N).max_in_branch.begin(),
                    merge_info->at(node2-N).max_in_branch.end());
                    if (merge_info->at(i).nfa < min(merge_info->at(node1-N).min_nfa_in_branch,
                                                    merge_info->at(node2-N).min_nfa_in_branch))
                        merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                    else
                        merge_info->at(i).min_nfa_in_branch = min(merge_info->at(node1-N).min_nfa_in_branch,
                                                                  merge_info->at(node2-N).min_nfa_in_branch);
                }

            } else {

                // one node is a "set" and the other is an indivisual sample: check merging condition
                if (node1>=N)
                {
                    if ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + 1) &&
                        (merge_info->at(i).nfa<merge_info->at(node1-N).min_nfa_in_branch))
                    {
                        merge_info->at(i).max_meaningful = true;
                        merge_info->at(i).max_in_branch.push_back(i);
                        merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;

                        for (int k =0; k<(int)merge_info->at(node1-N).max_in_branch.size(); k++)
                            merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;

                    } else {
                        merge_info->at(i).max_meaningful = false;
                        merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                        merge_info->at(node1-N).max_in_branch.begin(),
                        merge_info->at(node1-N).max_in_branch.end());
                        merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,
                                                                  merge_info->at(node1-N).min_nfa_in_branch);
                    }
                } else {
                    if ((merge_info->at(i).nfa < merge_info->at(node2-N).nfa + 1) &&
                        (merge_info->at(i).nfa<merge_info->at(node2-N).min_nfa_in_branch))
                    {
                        merge_info->at(i).max_meaningful = true;
                        merge_info->at(i).max_in_branch.push_back(i);
                        merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
                        for (int k =0; k<(int)merge_info->at(node2-N).max_in_branch.size(); k++)
                            merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
                    } else {
                        merge_info->at(i).max_meaningful = false;
                        merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),
                        merge_info->at(node2-N).max_in_branch.begin(),
                        merge_info->at(node2-N).max_in_branch.end());
                        merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,
                        merge_info->at(node2-N).min_nfa_in_branch);
                    }
                }
            }
        }
    }

    for (int i=0; i<(int)merge_info->size(); i++)
    {
        if (merge_info->at(i).max_meaningful)
        {
            vector<int> cluster;
            for (int k=0; k<(int)merge_info->at(i).elements.size();k++)
                cluster.push_back(merge_info->at(i).elements.at(k));
            meaningful_clusters->push_back(cluster);
        }
    }

}

int MaxMeaningfulClustering::nfa(float sigma, int k, int N)
{
    // use an approximation for the nfa calculations (faster)
    return -1*(int)NFA( N, k, (double) sigma, 0);
}

void accumulate_evidence(vector<vector<int> > *meaningful_clusters, int grow, Mat *co_occurrence);

void accumulate_evidence(vector<vector<int> > *meaningful_clusters, int grow, Mat *co_occurrence)
{
    for (int k=0; k<(int)meaningful_clusters->size(); k++)
        for (int i=0; i<(int)meaningful_clusters->at(k).size(); i++)
            for (int j=i; j<(int)meaningful_clusters->at(k).size(); j++)
                if (meaningful_clusters->at(k).at(i) != meaningful_clusters->at(k).at(j))
                {
                    co_occurrence->at<double>(meaningful_clusters->at(k).at(i), meaningful_clusters->at(k).at(j)) += grow;
                    co_occurrence->at<double>(meaningful_clusters->at(k).at(j), meaningful_clusters->at(k).at(i)) += grow;
                }
}

// ERFeatures structure stores additional features for a given ERStat instance
struct ERFeatures
{
    int area;
    Point center;
    Rect  rect;
    float intensity_mean;  ///< mean intensity of the whole region
    float intensity_std;  ///< intensity standard deviation of the whole region
    float boundary_intensity_mean;  ///< mean intensity of the boundary of the region
    float boundary_intensity_std;  ///< intensity standard deviation of the boundary of the region
    double stroke_mean;  ///< mean stroke width approximation of the whole region
    double stroke_std;  ///< stroke standard deviation of the whole region
    double gradient_mean;  ///< mean gradient magnitude of the whole region
    double gradient_std;  ///< gradient magnitude standard deviation of the whole region
};

float extract_features(InputOutputArray src, vector<ERStat> &regions, vector<ERFeatures> &features);
void  ergrouping(InputOutputArray src, vector<ERStat> &regions);

float extract_features(InputOutputArray src, vector<ERStat> &regions, vector<ERFeatures> &features)
{
    // assert correct image type
    CV_Assert( (src.type() == CV_8UC1) || (src.type() == CV_8UC3) );

    CV_Assert( !regions.empty() );
    CV_Assert( features.empty() );

    Mat grey = src.getMat();

    Mat gradient_magnitude = Mat_<float>(grey.size());
    get_gradient_magnitude( grey, gradient_magnitude);

    Mat region_mask = Mat::zeros(grey.rows+2, grey.cols+2, CV_8UC1);

    float max_stroke = 0;

    for (int r=0; r<(int)regions.size(); r++)
    {
        ERFeatures f;
        ERStat *stat = &regions.at(r);

        f.area = stat->area;
        f.rect = stat->rect;
        f.center = Point(f.rect.x+(f.rect.width/2),f.rect.y+(f.rect.height/2));

        if (regions.at(r).parent != NULL)
        {

            //Fill the region and calculate features
            Mat region = region_mask(Rect(Point(stat->rect.x,stat->rect.y),
                                          Point(stat->rect.br().x+2,stat->rect.br().y+2)));
            region = Scalar(0);
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            Rect rect;

			Mat m = Mat(grey(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x,stat->rect.br().y))));
            floodFill( m,
                       region, Point(stat->pixel%grey.cols - stat->rect.x, stat->pixel/grey.cols - stat->rect.y),
                       Scalar(255), &rect, Scalar(stat->level), Scalar(0), flags );
            rect.width += 2;
            rect.height += 2;
            Mat rect_mask = region_mask(Rect(stat->rect.x+1,stat->rect.y+1,stat->rect.width,stat->rect.height));


            Scalar mean,std;
            meanStdDev( grey(stat->rect), mean, std, rect_mask);
            f.intensity_mean = (float)mean[0];
            f.intensity_std  = (float)std[0];

            Mat tmp;
            distanceTransform(rect_mask, tmp, CV_DIST_L1,3);

            meanStdDev(tmp,mean,std,rect_mask);
            f.stroke_mean = mean[0];
            f.stroke_std  = std[0];

            if (f.stroke_mean > max_stroke)
                max_stroke = (float)f.stroke_mean;

            Mat element = getStructuringElement( MORPH_RECT, Size(5, 5), Point(2, 2) );
            dilate(rect_mask, tmp, element);
            absdiff(tmp, rect_mask, tmp);

            meanStdDev( grey(stat->rect), mean, std, tmp);
            f.boundary_intensity_mean = (float)mean[0];
            f.boundary_intensity_std  = (float)std[0];

            Mat tmp2;
            dilate(rect_mask, tmp, element);
            erode (rect_mask, tmp2, element);
            absdiff(tmp, tmp2, tmp);

            meanStdDev( gradient_magnitude(stat->rect), mean, std, tmp);
            f.gradient_mean = mean[0];
            f.gradient_std  = std[0];

            rect_mask = Scalar(0);

        } else {

            f.intensity_mean = 0;
            f.intensity_std  = 0;

            f.stroke_mean = 0;
            f.stroke_std  = 0;

            f.boundary_intensity_mean = 0;
            f.boundary_intensity_std  = 0;

            f.gradient_mean = 0;
            f.gradient_std  = 0;
        }

        features.push_back(f);
    }

    return max_stroke;
}

static bool edge_comp (Vec4f i,Vec4f j)
{
    Point a = Point(cvRound(i[0]), cvRound(i[1]));
    Point b = Point(cvRound(i[2]), cvRound(i[3]));
    double edist_i = cv::norm(a-b);
    a = Point(cvRound(j[0]), cvRound(j[1]));
    b = Point(cvRound(j[2]), cvRound(j[3]));
    double edist_j = cv::norm(a-b);
    return (edist_i<edist_j);
}

static bool find_vertex(vector<Point> &vertex, Point &p)
{
    for (int i=0; i<(int)vertex.size(); i++)
    {
        if (vertex.at(i) == p)
            return true;
    }
    return false;
}


/*!
    Find groups of Extremal Regions that are organized as text blocks. This function implements
    the grouping algorithm described in:
    Gomez L. and Karatzas D.: Multi-script Text Extraction from Natural Scenes, ICDAR 2013.
    Notice that this implementation constrains the results to horizontally-aligned text and
    latin script (since ERFilter default classifiers are trained only for latin script detection).

    The algorithm combines two different clustering techniques in a single parameter-free procedure
    to detect groups of regions organized as text. The maximally meaningful groups are fist detected
    in several feature spaces, where each feature space is a combination of proximity information
    (x,y coordinates) and a similarity measure (intensity, color, size, gradient magnitude, etc.),
    thus providing a set of hypotheses of text groups. Evidence Accumulation framework is used to
    combine all these hypotheses to get the final estimate. Each of the resulting groups are finally
    validated by a classifier in order to assest if they form a valid horizontally-aligned text block.

    \param  src            Vector of sinle channel images CV_8UC1 from wich the regions were extracted.
    \param  regions        Vector of ER's retreived from the ERFilter algorithm from each channel
    \param  filename       The XML or YAML file with the classifier model (e.g. trained_classifier_erGrouping.xml)
    \param  minProbability The minimum probability for accepting a group
    \param  groups         The output of the algorithm are stored in this parameter as list of rectangles.
*/
void erGrouping(InputArrayOfArrays _src, vector<vector<ERStat> > &regions, const std::string& filename, float minProbability, std::vector<Rect > &text_boxes)
{

    // TODO assert correct vector<Mat>

    CvBoost group_boost;
    if (ifstream(filename.c_str()))
        group_boost.load( filename.c_str(), "boost" );
    else
        CV_Error(CV_StsBadArg, "erGrouping: Default classifier file not found!");

    std::vector<Mat> src;
    _src.getMatVector(src);

    CV_Assert ( !src.empty() );
    CV_Assert ( src.size() == regions.size() );

    if (!text_boxes.empty())
    {
        text_boxes.clear();
    }

    for (int c=0; c<(int)src.size(); c++)
    {
        Mat img = src.at(c);

        // assert correct image type
        CV_Assert( img.type() == CV_8UC1 );

        CV_Assert( !regions.at(c).empty() );

        if ( regions.at(c).size() < 3 )
          continue;


        std::vector<vector<int> > meaningful_clusters;
        vector<ERFeatures> features;
        float max_stroke = extract_features(img,regions.at(c),features);

        MaxMeaningfulClustering   mm_clustering(METHOD_METR_SINGLE, METRIC_SEUCLIDEAN);

        Mat co_occurrence_matrix = Mat::zeros((int)regions.at(c).size(), (int)regions.at(c).size(), CV_64F);

        int num_features = MAX_NUM_FEATURES;

        // Find the Max. Meaningful Clusters in each feature space independently
        int dims[MAX_NUM_FEATURES] = {3,3,3,3,3,3,3,3,3};

        for (int f=0; f<num_features; f++)
        {
            unsigned int N = (unsigned int)regions.at(c).size();
            if (N<3) break;
            int dim = dims[f];
            double *data = (double*)malloc(dim*N * sizeof(double));
            if (data == NULL)
                CV_Error(CV_StsNoMem, "Not enough Memory for erGrouping hierarchical clustering structures!");
            int count = 0;
            for (int i=0; i<(int)regions.at(c).size(); i++)
            {
                data[count] = (double)features.at(i).center.x/img.cols;
                data[count+1] = (double)features.at(i).center.y/img.rows;
                switch (f)
                {
                    case 0:
                        data[count+2] = (double)features.at(i).intensity_mean/255;
                        break;
                    case 1:
                        data[count+2] = (double)features.at(i).boundary_intensity_mean/255;
                        break;
                    case 2:
                        data[count+2] = (double)features.at(i).rect.y/img.rows;
                        break;
                    case 3:
                        data[count+2] = (double)(features.at(i).rect.y+features.at(i).rect.height)/img.rows;
                        break;
                    case 4:
                        data[count+2] = (double)max(features.at(i).rect.height,
                                                    features.at(i).rect.width)/max(img.rows,img.cols);
                        break;
                    case 5:
                        data[count+2] = (double)features.at(i).stroke_mean/max_stroke;
                        break;
                    case 6:
                        data[count+2] = (double)features.at(i).area/(img.rows*img.cols);
                        break;
                    case 7:
                        data[count+2] = (double)(features.at(i).rect.height*
                                                 features.at(i).rect.width)/(img.rows*img.cols);
                        break;
                    case 8:
                        data[count+2] = (double)features.at(i).gradient_mean/255;
                        break;
                }
                count = count+dim;
            }

            mm_clustering(data, N, dim, METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, &meaningful_clusters);

            // Accumulate evidence in the coocurrence matrix
            accumulate_evidence(&meaningful_clusters, 1, &co_occurrence_matrix);

            free(data);
            meaningful_clusters.clear();
        }

        double minVal;
        double maxVal;
        minMaxLoc(co_occurrence_matrix, &minVal, &maxVal);

        maxVal = num_features - 1;
        minVal=0;

        co_occurrence_matrix = maxVal - co_occurrence_matrix;
        co_occurrence_matrix = co_occurrence_matrix / maxVal;

        // we want a sparse matrix
        double *D = (double*)malloc((regions.at(c).size()*regions.at(c).size()) * sizeof(double));
        if (D == NULL)
            CV_Error(CV_StsNoMem, "Not enough Memory for erGrouping hierarchical clustering structures!");

        int pos = 0;
        for (int i = 0; i<co_occurrence_matrix.rows; i++)
        {
            for (int j = i+1; j<co_occurrence_matrix.cols; j++)
            {
                D[pos] = (double)co_occurrence_matrix.at<double>(i, j);
                pos++;
            }
        }

        // Find the Max. Meaningful Clusters in the co-occurrence matrix
        mm_clustering(D, (unsigned int)regions.at(c).size(), METHOD_METR_AVERAGE, &meaningful_clusters);
        free(D);



        /* --------------------------------- Groups Validation --------------------------------*/
        /* Examine each of the clusters in order to assest if they are valid text lines or not */
        /* ------------------------------------------------------------------------------------*/

        vector<vector<float> > data_arrays(meaningful_clusters.size());
        vector<Rect> groups_rects(meaningful_clusters.size());

        // Collect group level features and classify the group
        for (int i=(int)meaningful_clusters.size()-1; i>=0; i--)
        {

            Rect group_rect;
            float sumx=0, sumy=0, sumxy=0, sumx2=0;

            // linear regression slope helps discriminating horizontal aligned groups
            for (int j=0; j<(int)meaningful_clusters.at(i).size();j++)
            {
                if (j==0)
                {
                    group_rect = regions.at(c).at(meaningful_clusters.at(i).at(j)).rect;
                } else {
                    group_rect = group_rect | regions.at(c).at(meaningful_clusters.at(i).at(j)).rect;
                }

                sumx  += regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.x +
                                    regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.width/2;
                sumy  += regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.y +
                                    regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.height/2;
                sumxy += (regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.x +
                                     regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.width/2)*
                         (regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.y +
                                     regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.height/2);
                sumx2 += (regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.x +
                                     regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.width/2)*
                         (regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.x +
                                     regions.at(c).at(meaningful_clusters.at(i).at(j)).rect.width/2);
            }
            // line coefficients
            //float a0=(sumy*sumx2-sumx*sumxy)/((int)meaningful_clusters.at(i).size()*sumx2-sumx*sumx);
            float a1=((int)meaningful_clusters.at(i).size()*sumxy-sumx*sumy) /
               ((int)meaningful_clusters.at(i).size()*sumx2-sumx*sumx);

            vector<float> data;
            if (a1 != a1)
                data_arrays.at(i).push_back(1.f);
            else
                data_arrays.at(i).push_back(a1);

            groups_rects.at(i) = group_rect;

            // group probability mean
            double group_probability_mean = 0;
            // number of non-overlapping regions
            vector<Rect> individual_components;

            // The variance of several similarity features is also helpful
            vector<float> strokes;
            vector<float> grad_magnitudes;
            vector<float> intensities;
            vector<float> bg_intensities;

            // We'll try to remove groups with repetitive patterns using averaged SAD
            // SAD = Sum of Absolute Differences
            Mat grey = img;
            Mat sad = Mat::zeros(regions.at(c).at(meaningful_clusters.at(i).at(0)).rect.size() , CV_8UC1);
            Mat region_mask = Mat::zeros(grey.rows+2, grey.cols+2, CV_8UC1);
            float sad_value = 0;
            Mat ratios = Mat::zeros(1, (int)meaningful_clusters.at(i).size(), CV_32FC1);
            //Mat holes  = Mat::zeros(1, (int)meaningful_clusters.at(i).size(), CV_32FC1);

            for (int j=0; j<(int)meaningful_clusters.at(i).size();j++)
            {
                ERStat *stat = &regions.at(c).at(meaningful_clusters.at(i).at(j));

                //Fill the region
                Mat region = region_mask(Rect(Point(stat->rect.x,stat->rect.y),
                                              Point(stat->rect.br().x+2,stat->rect.br().y+2)));
                region = Scalar(0);
                int newMaskVal = 255;
                int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
                Rect rect;

				Mat m = Mat(grey(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x,stat->rect.br().y))));
                floodFill( m,
                           region, Point(stat->pixel%grey.cols - stat->rect.x, stat->pixel/grey.cols - stat->rect.y),
                           Scalar(255), &rect, Scalar(stat->level), Scalar(0), flags );

                Mat mask = Mat::zeros(regions.at(c).at(meaningful_clusters.at(i).at(0)).rect.size() , CV_8UC1);
                resize(region, mask, mask.size());
                mask = mask - 254;
                if (j!=0)
                {
                    // accumulate Sum of Absolute Differences
                    absdiff(sad, mask, sad);
                    Scalar s = sum(sad);
                    sad_value += (float)s[0]/(sad.rows*sad.cols);
                }
                mask.copyTo(sad);
                ratios.at<float>(0,j) = (float)min(stat->rect.width, stat->rect.height) /
                                               max(stat->rect.width, stat->rect.height);
                //holes.at<float>(0,j) = (float)stat->hole_area_ratio;

                strokes.push_back((float)features.at(meaningful_clusters.at(i).at(j)).stroke_mean);
                grad_magnitudes.push_back((float)features.at(meaningful_clusters.at(i).at(j)).gradient_mean);
                intensities.push_back(features.at(meaningful_clusters.at(i).at(j)).intensity_mean);
                bg_intensities.push_back(features.at(meaningful_clusters.at(i).at(j)).boundary_intensity_mean);
                group_probability_mean += regions.at(c).at(meaningful_clusters.at(i).at(j)).probability;

                if (j==0)
                {
                    group_rect = features.at(meaningful_clusters.at(i).at(j)).rect;
                    individual_components.push_back(group_rect);
                } else {
                    bool matched = false;
                    for (int k=0; k<(int)individual_components.size(); k++)
                    {
                        Rect intersection = individual_components.at(k) &
                                            features.at(meaningful_clusters.at(i).at(j)).rect;

                        if ((intersection == features.at(meaningful_clusters.at(i).at(j)).rect) ||
                            (intersection == individual_components.at(k)))
                        {
                            individual_components.at(k) = individual_components.at(k) |
                                                          features.at(meaningful_clusters.at(i).at(j)).rect;
                            matched = true;
                        }
                    }

                    if (!matched)
                        individual_components.push_back(features.at(meaningful_clusters.at(i).at(j)).rect);

                    group_rect = group_rect | features.at(meaningful_clusters.at(i).at(j)).rect;
                }
            }
            group_probability_mean = group_probability_mean / meaningful_clusters.at(i).size();

            data_arrays.at(i).insert(data_arrays.at(i).begin(),(float)individual_components.size());

            // variance of widths and heights help to discriminate groups with high height variability
            vector<int> widths;
            vector<int> heights;
            // the MST edge orientations histogram may be dominated by the horizontal axis orientation
            Subdiv2D subdiv(Rect(0,0,src.at(0).cols,src.at(0).rows));

            for (int r=0; r < (int)individual_components.size(); r++)
            {
                widths.push_back(individual_components.at(r).width);
                heights.push_back(individual_components.at(r).height);

                Point2f fp( (float)individual_components.at(r).x + individual_components.at(r).width/2,
                            (float)individual_components.at(r).y + individual_components.at(r).height/2 );
                subdiv.insert(fp);
            }

            Scalar mean, std;
            meanStdDev(Mat(widths), mean, std);
            data_arrays.at(i).push_back((float)(std[0]/mean[0]));
            data_arrays.at(i).push_back((float)mean[0]);
            meanStdDev(Mat(heights), mean, std);
            data_arrays.at(i).push_back((float)(std[0]/mean[0]));

            vector<Vec4f> edgeList;
            subdiv.getEdgeList(edgeList);
            std::sort (edgeList.begin(), edgeList.end(), edge_comp);
            vector<Point> mst_vertices;

            int horiz_edges = 0, non_horiz_edges = 0;
            vector<float> edge_distances;

            for( size_t k = 0; k < edgeList.size(); k++ )
            {
                Vec4f e = edgeList[k];
                Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
                Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
                if (((pt0.x>0)&&(pt0.x<src.at(0).cols)&&(pt0.y>0)&&(pt0.y<src.at(0).rows) &&
                     (pt1.x>0)&&(pt1.x<src.at(0).cols)&&(pt1.y>0)&&(pt1.y<src.at(0).rows)) &&
                    ((!find_vertex(mst_vertices,pt0)) ||
                     (!find_vertex(mst_vertices,pt1))))
                {
                    double angle = atan2((double)(pt0.y-pt1.y),(double)(pt0.x-pt1.x));
                    //if ( (abs(angle) < 0.35) || (abs(angle) > 5.93) || ((abs(angle) > 2.79)&&(abs(angle) < 3.49)) )
                    if ( (std::abs(angle) < 0.25) || (std::abs(angle) > 6.03) || ((std::abs(angle) > 2.88)&&(std::abs(angle) < 3.4)) )
                    {
                        horiz_edges++;
                        edge_distances.push_back((float)norm(pt0-pt1));
                    }
                    else
                        non_horiz_edges++;
                    mst_vertices.push_back(pt0);
                    mst_vertices.push_back(pt1);
                }
            }

            if (horiz_edges == 0)
                data_arrays.at(i).push_back(0.f);
            else
                data_arrays.at(i).push_back((float)horiz_edges/(horiz_edges+non_horiz_edges));

            // remove groups where objects are not equidistant enough
            Scalar dist_mean, dist_std;
            meanStdDev(Mat(edge_distances),dist_mean, dist_std);
            if (dist_std[0] == 0)
                data_arrays.at(i).push_back(0.f);
            else
                data_arrays.at(i).push_back((float)(dist_std[0]/dist_mean[0]));

            if (dist_mean[0] == 0)
                data_arrays.at(i).push_back(0.f);
            else
                data_arrays.at(i).push_back((float)dist_mean[0]/data_arrays.at(i).at(3));

            //meanStdDev( holes, mean, std);
            //float holes_mean = (float)mean[0];
            meanStdDev( ratios, mean, std);

            data_arrays.at(i).push_back((float)sad_value / ((int)meaningful_clusters.at(i).size()-1));
            meanStdDev( Mat(strokes), mean, std);
            data_arrays.at(i).push_back((float)(std[0]/mean[0]));
            meanStdDev( Mat(grad_magnitudes), mean, std);
            data_arrays.at(i).push_back((float)(std[0]/mean[0]));
            meanStdDev( Mat(intensities), mean, std);
            data_arrays.at(i).push_back((float)std[0]);
            meanStdDev( Mat(bg_intensities), mean, std);
            data_arrays.at(i).push_back((float)std[0]);

            // Validate only groups with more than 2 non-overlapping regions
            if (data_arrays.at(i).at(0) > 2)
            {
                data_arrays.at(i).insert(data_arrays.at(i).begin(),0.f);
                float votes = group_boost.predict( Mat(data_arrays.at(i)), Mat(), Range::all(), false, true );
                // Logistic Correction returns a probability value (in the range(0,1))
                double probability = (double)1-(double)1/(1+exp(-2*votes));

                if (probability > minProbability)
                    text_boxes.push_back(groups_rects.at(i));
            }
        }

    }

    // check for colinear groups that can be merged
    for (int i=0; i<(int)text_boxes.size(); i++)
    {
        int ay1 = text_boxes.at(i).y;
        int ay2 = text_boxes.at(i).y + text_boxes.at(i).height;
        int ax1 = text_boxes.at(i).x;
        int ax2 = text_boxes.at(i).x + text_boxes.at(i).width;
        for (int j=(int)text_boxes.size()-1; j>i; j--)
        {
            int by1 = text_boxes.at(j).y;
            int by2 = text_boxes.at(j).y + text_boxes.at(j).height;
            int bx1 = text_boxes.at(j).x;
            int bx2 = text_boxes.at(j).x + text_boxes.at(j).width;

            int y_intersection = min(ay2,by2) - max(ay1,by1);

            if (y_intersection > 0.6*(max(text_boxes.at(i).height,text_boxes.at(j).height)))
            {
                int xdist = min(std::abs(ax2-bx1),std::abs(bx2-ax1));
                Rect intersection  = text_boxes.at(i) & text_boxes.at(j);
                if ( (xdist < 0.75*(max(text_boxes.at(i).height,text_boxes.at(j).height))) ||
                     (intersection.width != 0))
                {
                    text_boxes.at(i) = text_boxes.at(i) | text_boxes.at(j);
                    text_boxes.erase(text_boxes.begin()+j);
                }
            }

        }
    }

}

