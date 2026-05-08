// node_graph_rcpp.cpp
// Faithful Rcpp port of stemCluster.create_node_graph() and
// crownCluster.create_node_graph() from TreeAIBox.
//
// For each component (identified by an integer label vector), this function
// finds the k nearest *component* neighbours by centroid distance, then
// computes the actual minimum point-to-point 3D distance and 2D XY distance
// between those component pairs (matching the cKDTree per-component approach
// in Python).  It also returns the 2D centroid distance so the caller can
// apply the stemCluster 2D-only filter or the crownCluster dual filter.
//
// Python reference:
//   stemCluster.create_node_graph():
//     - nComp-wise centroid
//     - k=20 nearest component centroids
//     - actual min 3D dist via scipy.spatial.cKDTree
//     - edge filter: nn_dist2d < max_distance  (2D only, stemCluster)
//   crownCluster.create_node_graph():  identical structure, same 2D filter
//
// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <set>

using namespace Rcpp;

// Simple inline squared-distance helpers
static inline double dist3sq(double ax, double ay, double az,
                              double bx, double by, double bz) {
    double dx = ax-bx, dy = ay-by, dz = az-bz;
    return dx*dx + dy*dy + dz*dz;
}
static inline double dist2sq(double ax, double ay,
                              double bx, double by) {
    double dx = ax-bx, dy = ay-by;
    return dx*dx + dy*dy;
}

//' Build component-level node graph with true minimum point-to-point distances
//'
//' Faithfully replicates \code{stemCluster.create_node_graph()} /
//' \code{crownCluster.create_node_graph()} from TreeAIBox.
//'
//' For each component, the k nearest component-neighbours (by centroid) are
//' found.  For each candidate pair, the actual minimum 3-D point-to-point
//' distance and the 2-D XY centroid-to-centroid distance are computed.
//' Edges where \code{dist2d >= max_dist_2d} are dropped.  The remaining
//' edges are returned so the R caller can build an igraph and run Dijkstra.
//'
//' @param xyz       numeric matrix (n_pts x 3).  Points of the decimated cloud.
//' @param labels    integer vector length n_pts.  0-based component label per
//'                  decimated point (e.g. from \code{cut_pursuit_l0_rcpp}).
//' @param k         number of nearest component-centroid neighbours to consider
//'                  per component (TreeAIBox default: 20).
//' @param max_dist_2d  2-D XY distance threshold.  Pairs farther than this are
//'                  dropped (TreeAIBox: \code{max_isolated_distance} = 0.3 m
//'                  for both stemCluster and crownCluster).
//' @return A list with elements:
//'   \item{from}{integer vector: 1-based source component node index}
//'   \item{to}{integer vector: 1-based target component node index}
//'   \item{dist3d}{numeric: true minimum 3-D point-to-point distance}
//'   \item{dist2d}{numeric: 2-D XY centroid distance}
//'   \item{n_comps}{integer: total number of unique components}
//' @export
// [[Rcpp::export]]
List create_node_graph_rcpp(
    NumericMatrix xyz,
    IntegerVector labels,
    int    k           = 20,
    double max_dist_2d = 0.3
) {
    const int n_pts = xyz.nrow();
    if (labels.size() != n_pts)
        stop("create_node_graph_rcpp: labels length must equal nrow(xyz).");

    // ---- 1. Map unique label values to 0-based indices ------------------
    std::vector<int> lbl(labels.begin(), labels.end());
    std::vector<int> unique_lbls = lbl;
    std::sort(unique_lbls.begin(), unique_lbls.end());
    unique_lbls.erase(std::unique(unique_lbls.begin(), unique_lbls.end()), unique_lbls.end());
    const int n_comps = (int)unique_lbls.size();

    // label value → 0-based component index
    std::vector<int> lbl2idx(n_comps);
    for (int c = 0; c < n_comps; ++c) lbl2idx[c] = c;   // identity after remap

    // Build a fast lookup: label value → component index
    // Using a sorted vector + lower_bound
    auto label_to_comp = [&](int lv) -> int {
        auto it = std::lower_bound(unique_lbls.begin(), unique_lbls.end(), lv);
        return (int)(it - unique_lbls.begin());
    };

    std::vector<int> comp_of(n_pts);
    for (int i = 0; i < n_pts; ++i)
        comp_of[i] = label_to_comp(lbl[i]);

    // ---- 2. Build per-component point lists and centroids ---------------
    std::vector<std::vector<int>> comp_pts(n_comps);
    for (int i = 0; i < n_pts; ++i)
        comp_pts[comp_of[i]].push_back(i);

    // Centroids (x, y, z)
    std::vector<double> cx(n_comps, 0.), cy(n_comps, 0.), cz(n_comps, 0.);
    for (int c = 0; c < n_comps; ++c) {
        int sz = (int)comp_pts[c].size();
        if (sz == 0) continue;
        for (int idx : comp_pts[c]) {
            cx[c] += xyz(idx, 0);
            cy[c] += xyz(idx, 1);
            cz[c] += xyz(idx, 2);
        }
        cx[c] /= sz;  cy[c] /= sz;  cz[c] /= sz;
    }

    // ---- 3. k-nearest component centroids (brute force; typically << 10k comps) --
    int k_eff = std::min(k, n_comps - 1);
    if (k_eff <= 0 || n_comps <= 1) {
        // Return empty graph
        return List::create(
            _["from"]    = IntegerVector(0),
            _["to"]      = IntegerVector(0),
            _["dist3d"]  = NumericVector(0),
            _["dist2d"]  = NumericVector(0),
            _["n_comps"] = n_comps
        );
    }

    // For each component we collect (neighbour_comp, centroid_dist2d) pairs
    // sorted by centroid_dist2d, take top k_eff.
    // We keep edges as an unordered set of pairs to avoid duplicates.
    // Output in 1-based (R) indexing.
    std::vector<int>    out_from, out_to;
    std::vector<double> out_d3,  out_d2;
    out_from.reserve(n_comps * k_eff);
    out_to.reserve(n_comps * k_eff);
    out_d3.reserve(n_comps * k_eff);
    out_d2.reserve(n_comps * k_eff);

    // Track which (min,max) pairs we've already processed
    std::set<std::pair<int,int>> seen;

    for (int c = 0; c < n_comps; ++c) {
        // Compute centroid 2D distances to all other components
        std::vector<std::pair<double, int>> d2_comp; // (dist2d, comp_index)
        d2_comp.reserve(n_comps - 1);
        for (int j = 0; j < n_comps; ++j) {
            if (j == c) continue;
            double d2 = std::sqrt(dist2sq(cx[c], cy[c], cx[j], cy[j]));
            d2_comp.push_back({d2, j});
        }
        // Partial sort: keep k_eff smallest
        std::partial_sort(d2_comp.begin(),
                          d2_comp.begin() + k_eff,
                          d2_comp.end());

        for (int ki = 0; ki < k_eff; ++ki) {
            double centroid_d2 = d2_comp[ki].first;
            int    nb           = d2_comp[ki].second;

            // Primary filter: 2D centroid distance
            if (centroid_d2 >= max_dist_2d) break; // sorted, rest only farther

            // Deduplicate undirected edges
            int ea = std::min(c, nb), eb = std::max(c, nb);
            if (!seen.insert({ea, eb}).second) continue;

            // Compute TRUE minimum 3D point-to-point distance between
            // the two components' actual point sets.
            // This mirrors Python: cKDTree(pts_b).query(pts_a)[0].min()
            double min3d_sq = std::numeric_limits<double>::infinity();
            for (int ia : comp_pts[c]) {
                double xa = xyz(ia, 0), ya = xyz(ia, 1), za = xyz(ia, 2);
                for (int ib : comp_pts[nb]) {
                    double d = dist3sq(xa, ya, za,
                                       xyz(ib,0), xyz(ib,1), xyz(ib,2));
                    if (d < min3d_sq) min3d_sq = d;
                }
            }
            double min3d = std::sqrt(min3d_sq);

            // 1-based output (R convention)
            out_from.push_back(c  + 1);
            out_to.push_back(nb + 1);
            out_d3.push_back(min3d);
            out_d2.push_back(centroid_d2);
        }
    }

    return List::create(
        _["from"]    = IntegerVector(out_from.begin(), out_from.end()),
        _["to"]      = IntegerVector(out_to.begin(),   out_to.end()),
        _["dist3d"]  = NumericVector(out_d3.begin(),   out_d3.end()),
        _["dist2d"]  = NumericVector(out_d2.begin(),   out_d2.end()),
        _["n_comps"] = n_comps
    );
}
