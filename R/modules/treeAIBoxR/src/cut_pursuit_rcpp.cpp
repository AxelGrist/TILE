// cut_pursuit_rcpp.cpp
// Rcpp wrapper for Landrieu's L0-cut-pursuit algorithm.
// Matches the interface called by crownCluster.py in TreeAIBox:
//   labels = _cut_pursuit.perform_cut_pursuit(
//       reg_strength=1.0, D=3, pc_vec=..., edge_weights=..., Eu=..., Ev=...,
//       verbose=False)
//
// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include "cut_pursuit/API.h"

//' Run L0 cut-pursuit graph partitioning
//'
//' Mirrors \code{crownCluster.init_cutpursuit()} from TreeAIBox.
//' Takes a point cloud (n_nodes x D observation matrix), a KNN edge list,
//' and returns a component label per node (0-based).
//'
//' @param obs        numeric matrix (n_nodes x D).  For CrownClustersSP, D=3
//'                   and the rows are the XYZ of each decimated point.
//' @param Eu         integer vector of 0-based edge source indices.
//' @param Ev         integer vector of 0-based edge target indices.
//' @param edge_weights numeric vector of edge weights (length n_edges).
//'                   Pass \code{rep(1, n_edges)} for uniform weights.
//' @param node_weights numeric vector of node weights (length n_nodes).
//'                   Pass \code{rep(1, n_nodes)} for uniform weights.
//' @param lambda     regularisation strength (TreeAIBox default: 1.0).
//' @param cutoff     minimum component size (0 = no minimum).
//' @param mode       fidelity: 1.0 = L2 (default for crown blobs).
//' @param speed      0=slow, 1=standard (default), 2=fast, 3=ludicrous.
//' @param weight_decay edge-weight decay (0 = no decay).
//' @param verbose    0 = silent.
//' @return integer vector length n_nodes: component index (0-based) per node.
//' @export
// [[Rcpp::export]]
Rcpp::IntegerVector cut_pursuit_l0_rcpp(
    Rcpp::NumericMatrix obs,
    Rcpp::IntegerVector Eu,
    Rcpp::IntegerVector Ev,
    Rcpp::NumericVector edge_weights,
    Rcpp::NumericVector node_weights,
    double lambda      = 1.0,
    int    cutoff      = 0,
    double mode        = 1.0,
    double speed       = 1.0,
    double weight_decay = 0.0,
    double verbose      = 0.0
) {
    const uint32_t n_nodes = static_cast<uint32_t>(obs.nrow());
    const uint32_t n_edges = static_cast<uint32_t>(Eu.size());
    const uint32_t nObs    = static_cast<uint32_t>(obs.ncol());

    if ((uint32_t)Ev.size() != n_edges)
        Rcpp::stop("cut_pursuit_l0_rcpp: Eu and Ev must have the same length.");
    if ((uint32_t)edge_weights.size() != n_edges)
        Rcpp::stop("cut_pursuit_l0_rcpp: edge_weights length must match Eu/Ev.");
    if ((uint32_t)node_weights.size() != n_nodes)
        Rcpp::stop("cut_pursuit_l0_rcpp: node_weights length must match n_nodes.");

    // --- Build flat observation array (row-major: node0d0,node0d1,...) ---
    // Rcpp::NumericMatrix is column-major, so we transpose here.
    std::vector<float> observation(n_nodes * nObs);
    for (uint32_t i = 0; i < n_nodes; ++i)
        for (uint32_t d = 0; d < nObs; ++d)
            observation[i * nObs + d] = static_cast<float>(obs(i, d));

    // --- Convert edge indices ---
    std::vector<uint32_t> eu(n_edges), ev(n_edges);
    for (uint32_t e = 0; e < n_edges; ++e) {
        eu[e] = static_cast<uint32_t>(Eu[e]);
        ev[e] = static_cast<uint32_t>(Ev[e]);
    }

    // --- Convert weights ---
    std::vector<float> ew(n_edges), nw(n_nodes);
    for (uint32_t e = 0; e < n_edges; ++e) ew[e] = static_cast<float>(edge_weights[e]);
    for (uint32_t i = 0; i < n_nodes; ++i) nw[i] = static_cast<float>(node_weights[i]);

    // --- Output buffers ---
    std::vector<float>    solution(n_nodes * nObs, 0.f);
    std::vector<uint32_t> in_component(n_nodes, 0);
    std::vector<std::vector<uint32_t>> components;

    // --- Call cut_pursuit (C-style segmentation variant) ---
    CP::cut_pursuit<float>(
        n_nodes,
        n_edges,
        nObs,
        observation.data(),
        eu.data(),
        ev.data(),
        ew.data(),
        nw.data(),
        solution.data(),
        in_component,
        components,
        static_cast<float>(lambda),
        static_cast<uint32_t>(cutoff),
        mode,
        speed,
        weight_decay,
        verbose
    );

    // Return in_component as 0-based integer labels
    Rcpp::IntegerVector result(n_nodes);
    for (uint32_t i = 0; i < n_nodes; ++i)
        result[i] = static_cast<int>(in_component[i]);
    return result;
}
