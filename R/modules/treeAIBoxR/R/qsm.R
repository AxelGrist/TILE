# qsm.R
#
# R port of NRCan TreeAIBox modules/qsm/applyQSM.py

# Suppress R CMD CHECK "no visible binding" notes for data.table NSE columns
utils::globalVariables(c(".", ".N", "x", "y", "z", "stem", "seg",
                         "cx", "cy", "cz", "stem_label", "count"))
# (Xi, Raguet 2019+; model weights CC BY-NC 4.0)
#
# Computes per-tree Quantitative Structure Models from wood-classified TLS
# point clouds.  The upstream Python pipeline (applyQSM.py) is reproduced
# faithfully:
#
#   1. initSegmentation : cut-pursuit over-segmentation of stem + branch pts
#   2. applyQSM         : connectivity graph → Dijkstra skeleton → radius fit
#
# The output cylinder table can be saved as RDS for later aRchi analysis.
#
# Dependencies (all already used elsewhere in the TILE pipeline):
#   treeisoR  -- cut_pursuit_segment (kNN cut-pursuit, Rcpp)
#   RANN      -- nn2 (fast kNN)
#   igraph    -- distances / shortest_paths (Dijkstra)
#   data.table -- fast group-by aggregation

# ---------------------------------------------------------------------------
# .circle_fit_kasa : linear algebraic circle fit (Kasa 1976).
# Robust for TLS point cross-sections with moderate noise.
# Returns list(xc, yc, r, sigma).
# ---------------------------------------------------------------------------
.circle_fit_kasa <- function(x, y) {
  n <- length(x)
  if (n < 4L) {
    xc <- mean(x); yc <- mean(y)
    r  <- max(median(sqrt((x - xc)^2 + (y - yc)^2)), 1e-4)
    return(list(xc = xc, yc = yc, r = r, sigma = r))
  }
  # Linear LS: [x, y, 1] * [2xc, 2yc, xc^2+yc^2-r^2]' = x^2+y^2
  A    <- cbind(x, y, 1)
  b    <- x^2 + y^2
  fit  <- tryCatch(lm.fit(A, b)$coefficients, error = function(e) NULL)
  if (is.null(fit) || any(!is.finite(fit))) {
    xc <- mean(x); yc <- mean(y)
    r  <- max(median(sqrt((x - xc)^2 + (y - yc)^2)), 1e-4)
    return(list(xc = xc, yc = yc, r = r, sigma = r))
  }
  xc  <- fit[1L] / 2
  yc  <- fit[2L] / 2
  r   <- sqrt(xc^2 + yc^2 + fit[3L])
  if (!is.finite(r) || r <= 0) {
    xc <- mean(x); yc <- mean(y)
    r  <- max(median(sqrt((x - xc)^2 + (y - yc)^2)), 1e-4)
    return(list(xc = xc, yc = yc, r = r, sigma = r))
  }
  res   <- abs(sqrt((x - xc)^2 + (y - yc)^2) - r)
  sigma <- median(res)
  list(xc = xc, yc = yc, r = max(r, 1e-4), sigma = sigma)
}

# ---------------------------------------------------------------------------
# .qsm_init_segmentation
# Ports applyQSM.initSegmentation().
# Over-segments stem and branch points separately via cut-pursuit.
# pts: matrix (n, 4) [x, y, z, stemcls]; stemcls >= 2 → stem.
# Returns integer vector (length n) of 0-based segment labels.
# ---------------------------------------------------------------------------
.qsm_init_segmentation <- function(pts,
                                    K_stem     = 20L,
                                    reg_stem   = 5.0,
                                    K_branch   = 3L,
                                    reg_branch = 0.01,
                                    threads    = 1L) {
  if (!requireNamespace("treeisoR", quietly = TRUE))
    stop(".qsm_init_segmentation requires treeisoR::cut_pursuit_segment(). ",
         "Rebuild treeisoR after the new cut_pursuit_segment_cpp export.")
  n          <- nrow(pts)
  segs       <- integer(n)
  stemcls    <- pts[, 4L]
  stem_idx   <- which(stemcls >= 2L)
  branch_idx <- which(stemcls <  2L)
  n_stem <- 0L
  if (length(stem_idx) >= 2L) {
    raw    <- treeisoR::cut_pursuit_segment(
                pts[stem_idx, 1:3, drop = FALSE],
                K = K_stem, reg_strength = reg_stem, threads = threads)
    remap  <- match(raw, unique(raw))
    n_stem <- max(remap)
    segs[stem_idx] <- remap - 1L   # 0-based
  }
  if (length(branch_idx) >= 2L) {
    raw   <- treeisoR::cut_pursuit_segment(
               pts[branch_idx, 1:3, drop = FALSE],
               K = K_branch, reg_strength = reg_branch, threads = threads)
    remap <- match(raw, unique(raw))
    segs[branch_idx] <- n_stem + remap - 1L
  }
  segs
}

# ---------------------------------------------------------------------------
# .qsm_seg_centroids
# Aggregate per-segment mean XYZ, max stemcls, and point count.
# Returns a data.table ordered by seg label (0-based).
# ---------------------------------------------------------------------------
.qsm_seg_centroids <- function(pts, init_segs) {
  dt  <- data.table::data.table(
    x    = pts[, 1L],
    y    = pts[, 2L],
    z    = pts[, 3L],
    stem = pts[, 4L],
    seg  = init_segs)
  agg <- dt[, .(cx = mean(x), cy = mean(y), cz = mean(z),
                stem_label = max(stem), count = .N), by = seg]
  data.table::setorder(agg, seg)
  agg
}

# ---------------------------------------------------------------------------
# .qsm_find_stems
# Ports findStems(): find the main-stem path as the longest Dijkstra route
# through stem-labelled segment centroids.
# Returns integer vector of 1-based row indices into segs_agg.
# ---------------------------------------------------------------------------
.qsm_find_stems <- function(segs_agg, k_nn = 5L, weight_min_dist = 0.05) {
  stem_rows <- which(segs_agg$stem_label >= 2L)
  if (length(stem_rows) < 2L) return(stem_rows)

  sc <- as.matrix(segs_agg[stem_rows, .(cx, cy, cz)])
  n  <- nrow(sc)
  kk <- min(k_nn, n - 1L)

  nn     <- RANN::nn2(sc, sc, k = kk + 1L)
  from_v <- rep(seq_len(n), each = kk)
  to_v   <- as.vector(t(nn$nn.idx[, -1L, drop = FALSE]))
  dist_v <- as.vector(t(nn$nn.dists[, -1L, drop = FALSE]))
  valid  <- to_v > 0L & from_v != to_v & is.finite(dist_v)
  from_v <- from_v[valid]; to_v <- to_v[valid]; dist_v <- dist_v[valid]

  # Squared weights prevent short-cutting around obtuse angles (upstream note)
  w <- (weight_min_dist^2) * dist_v^2

  g <- igraph::graph_from_data_frame(
    data.frame(from = from_v, to = to_v, weight = w),
    directed = FALSE,
    vertices = data.frame(name = seq_len(n)))

  start_i <- which.min(sc[, 3L])          # lowest Z = ground level
  dists   <- igraph::distances(g, v = start_i, mode = "all")[1L, ]
  conn    <- which(is.finite(dists))
  if (length(conn) == 0L) return(stem_rows[start_i])
  end_i   <- conn[which.max(dists[conn])]

  path <- igraph::shortest_paths(g, from = start_i, to = end_i,
                                  mode = "all", output = "vpath")$vpath[[1L]]
  stem_rows[as.integer(path)]
}

# ---------------------------------------------------------------------------
# .qsm_seg_adjacency
# Ports getConnectivity(): find cross-segment point pairs within max_distance.
# Returns integer matrix (m, 2) of adjacent (sorted) segment-row-index pairs.
# ---------------------------------------------------------------------------
.qsm_seg_adjacency <- function(pts, init_segs, k = 6L, max_distance = 0.03) {
  n  <- nrow(pts)
  kk <- min(k, n - 1L)
  nn <- RANN::nn2(pts[, 1:3, drop = FALSE], pts[, 1:3, drop = FALSE],
                  k = kk + 1L, searchtype = "radius", radius = max_distance)
  from_v <- rep(seq_len(n), each = kk)
  to_v   <- as.vector(t(nn$nn.idx[, -1L, drop = FALSE]))
  valid  <- to_v > 0L & from_v != to_v
  from_v <- from_v[valid]; to_v <- to_v[valid]

  cross <- init_segs[from_v] != init_segs[to_v]
  if (!any(cross)) return(matrix(integer(0), 0L, 2L))

  raw <- cbind(init_segs[from_v[cross]], init_segs[to_v[cross]])
  raw <- t(apply(raw, 1L, sort))           # normalise order
  unique(raw)                              # unique adjacent pairs
}

# ---------------------------------------------------------------------------
# .qsm_segment_graph
# Ports getPointwiseClusterDistance(): weighted igraph over segment centroids.
# Edges between adjacent segments (point-level) get small squared weights;
# edges to non-adjacent segments beyond occlusion_cutoff are removed (Inf).
# ---------------------------------------------------------------------------
.qsm_segment_graph <- function(segs_agg, adj_pairs,
                                k_nn             = 6L,
                                occlusion_cutoff = 0.4,
                                weight_min_dist  = 0.03) {
  sc    <- as.matrix(segs_agg[, .(cx, cy, cz)])
  n     <- nrow(sc)
  kk    <- min(k_nn, n - 1L)
  is_st <- segs_agg$stem_label >= 2L

  nn     <- RANN::nn2(sc, sc, k = kk + 1L)
  from_v <- rep(seq_len(n), each = kk)
  to_v   <- as.vector(t(nn$nn.idx[, -1L, drop = FALSE]))
  dist_v <- as.vector(t(nn$nn.dists[, -1L, drop = FALSE]))
  valid  <- to_v > 0L & from_v != to_v & is.finite(dist_v)
  from_v <- from_v[valid]; to_v <- to_v[valid]; dist_v <- dist_v[valid]

  # Drop stem↔stem edges: they are resolved by findStems already
  stem_stem <- is_st[from_v] & is_st[to_v]
  from_v <- from_v[!stem_stem]; to_v <- to_v[!stem_stem]; dist_v <- dist_v[!stem_stem]
  if (length(from_v) == 0L) {
    return(igraph::make_empty_graph(n = n, directed = FALSE))
  }

  # Build adjacency lookup (using 0-based seg labels stored in segs_agg$seg)
  seg_of <- segs_agg$seg          # 0-based label for each row
  if (!is.null(adj_pairs) && nrow(adj_pairs) > 0L) {
    adj_key <- paste(adj_pairs[, 1L], adj_pairs[, 2L], sep = "_")
    pair_key <- paste(pmin(seg_of[from_v], seg_of[to_v]),
                      pmax(seg_of[from_v], seg_of[to_v]), sep = "_")
    connected <- pair_key %in% adj_key
  } else {
    connected <- rep(FALSE, length(from_v))
  }

  w <- ifelse(connected,
              (weight_min_dist^2) * dist_v^2,
              dist_v)                              # raw distance for unconfirmed
  w[dist_v > occlusion_cutoff * 0.5] <- Inf       # cut long-range occlusions
  ok <- is.finite(w)

  # Safety: if nothing survived, fall back to raw-distance graph
  if (!any(ok)) { w[] <- dist_v; ok <- is.finite(w) }

  igraph::graph_from_data_frame(
    data.frame(from = from_v[ok], to = to_v[ok], weight = w[ok]),
    directed = FALSE,
    vertices = data.frame(name = seq_len(n)))
}

# ---------------------------------------------------------------------------
# .qsm_recreate_tree_path
# Ports recreateTreePath(): Dijkstra from stem nodes to all segments,
# then traces branch paths root→tip.
# Returns list(tree, segs_labels, n_branches).
# ---------------------------------------------------------------------------
.qsm_recreate_tree_path <- function(stem_seg_idx, graph, n_segs,
                                     max_graph_distance = 40,
                                     init_seg_id        = 2L,
                                     segs_labels        = NULL) {
  if (is.null(segs_labels)) {
    segs_labels <- integer(n_segs)
    segs_labels[stem_seg_idx] <- 1L
  }
  tree <- list(stem_seg_idx)   # path 1 = main stem

  # Distances from every stem node to every segment
  dmat <- igraph::distances(graph, v = stem_seg_idx, mode = "all")
  # dmat rows = stem nodes, cols = all segments (1..n_segs)
  min_dist  <- apply(dmat, 2L, min)
  nearest_s <- apply(dmat, 2L, which.min)   # index into stem_seg_idx

  used <- logical(n_segs)
  used[stem_seg_idx] <- TRUE

  # Process segments farthest-first (mirrors Python sorted_indices[::-1])
  ord <- order(min_dist, decreasing = TRUE)
  seg_id <- init_seg_id

  for (i in ord) {
    if (used[i]) next
    if (!is.finite(min_dist[i]) || min_dist[i] > max_graph_distance) next

    target_v <- stem_seg_idx[nearest_s[i]]
    path_res <- igraph::shortest_paths(graph, from = i, to = target_v,
                                        mode    = "all",
                                        output  = "vpath")$vpath[[1L]]
    if (length(path_res) == 0L) next
    path_v    <- as.integer(rev(path_res))  # root→tip
    new_nodes <- path_v[!used[path_v]]
    if (length(new_nodes) == 0L) next

    used[new_nodes] <- TRUE
    tree[[length(tree) + 1L]] <- path_v
    segs_labels[new_nodes[new_nodes != target_v]] <- seg_id
    seg_id <- seg_id + 1L
  }
  list(tree = tree, segs_labels = segs_labels, n_branches = seg_id - 1L)
}

# ---------------------------------------------------------------------------
# .qsm_clean_tree
# Ports cleanTree(): remove branch paths whose tip segment has < min_pts pts.
# ---------------------------------------------------------------------------
.qsm_clean_tree <- function(tree, seg_counts, segs_labels, min_pts = 5L) {
  keep <- vapply(tree, function(path) {
    tip <- path[length(path)]
    if (length(path) < 3L) return(seg_counts[tip] > min_pts)
    TRUE
  }, logical(1L))
  for (i in which(!keep))
    segs_labels[tree[[i]][-1L]] <- 0L
  list(tree = tree[keep], segs_labels = segs_labels)
}

# ---------------------------------------------------------------------------
# .qsm_calculate_radius
# Ports calculateRadius(): algebraic circle fit per segment node.
# Projects points onto the plane perpendicular to the local branch direction,
# fits a circle, applies a 3-point running median filter on radii.
# ---------------------------------------------------------------------------
.qsm_calculate_radius <- function(pts, tree, segs_agg, init_segs,
                                   min_r = 0.04, verbose = FALSE) {
  # seg_to_pts: list keyed by string(0-based seg label) → point row indices
  seg_to_pts <- split(seq_len(nrow(pts)), as.character(init_segs))

  n_br <- length(tree)
  .pb <- if (verbose && n_br > 1L) {
    function(i) {
      filled <- round(30L * i / n_br)
      cat(sprintf("\r  [%s] %3d%%  (%d/%d branches)",
                  paste0(strrep("=", filled), strrep(" ", 30L - filled)),
                  round(100L * i / n_br), i, n_br),
          file = stderr())
      if (i == n_br) cat("\n", file = stderr())
    }
  } else function(i) invisible(NULL)

  tree_cr <- vector("list", n_br)
  for (bi in seq_along(tree)) {
    .pb(bi)
    path <- tree[[bi]]
    if (length(path) < 2L) { tree_cr[[bi]] <- list(); next }

    # Directional vectors along each segment edge
    seg_dir <- matrix(0, length(path) - 1L, 3L)
    for (si in seq_len(length(path) - 1L)) {
      v <- c(segs_agg$cx[path[si+1L]] - segs_agg$cx[path[si]],
             segs_agg$cy[path[si+1L]] - segs_agg$cy[path[si]],
             segs_agg$cz[path[si+1L]] - segs_agg$cz[path[si]])
      len <- sqrt(sum(v^2))
      seg_dir[si, ] <- if (len > 1e-8) v / len else c(0, 0, 1)
    }

    path_cr <- vector("list", length(path) - 1L)
    for (si in seq_len(length(path) - 1L)) {
      node    <- path[si + 1L]
      seg_lbl <- as.character(segs_agg$seg[node])
      idx     <- seg_to_pts[[seg_lbl]]
      cen     <- c(segs_agg$cx[node], segs_agg$cy[node], segs_agg$cz[node])

      if (is.null(idx) || length(idx) < 5L) {
        path_cr[[si]] <- c(cen, min_r)
        next
      }
      seg_pts <- pts[idx, 1:3, drop = FALSE]
      d       <- sweep(seg_pts, 2L, cen)
      # Project out the axial component; fit circle in the remaining 2D plane
      proj  <- as.numeric(d %*% seg_dir[si, ])
      d2    <- d - outer(proj, seg_dir[si, ])
      r0    <- median(sqrt(rowSums(d2^2)))
      cf    <- .circle_fit_kasa(d2[, 1L], d2[, 2L])
      r     <- if (is.finite(cf$sigma) && cf$sigma / max(cf$r, 1e-6) <= 0.3)
                 min(cf$r, r0 * 1.2)
               else r0
      path_cr[[si]] <- c(cen, max(r, min_r))
    }

    # Prepend root node (same radius as first segment)
    r0       <- if (length(path_cr) > 0L) path_cr[[1L]][4L] else min_r
    root_node <- path[1L]
    root_cr  <- c(segs_agg$cx[root_node], segs_agg$cy[root_node],
                  segs_agg$cz[root_node], r0)
    path_cr  <- c(list(root_cr), path_cr)

    # 3-point running median on radii (matches upstream medfilt kernel_size=3)
    if (length(path_cr) >= 3L) {
      radii <- vapply(path_cr, `[[`, numeric(1L), 4L)
      radii <- as.numeric(stats::runmed(radii, k = 3L))
      for (ni in seq_along(path_cr))
        path_cr[[ni]][4L] <- max(radii[ni], min_r)
    }
    tree_cr[[bi]] <- path_cr
  }
  tree_cr
}

# ---------------------------------------------------------------------------
# PUBLIC: apply_qsm
# ---------------------------------------------------------------------------

#' Apply a QSM to a wood-classified per-tree point cloud.
#'
#' R port of \code{applyQSM.applyQSM()} from NRCan TreeAIBox
#' (Xi et al. 2023, CC BY-NC 4.0).
#'
#' @section Input requirements:
#' \code{pts} must be a matrix with at least 4 columns:
#' \enumerate{
#'   \item X (m)
#'   \item Y (m)
#'   \item Z (m)
#'   \item stemcls: integer, \code{>= 2} = trunk/stem point (from StemCls
#'         model); \code{< 2} = branch.  Points with \code{WoodLabel == 1}
#'         (foliage from WoodCls) should be filtered out **before** calling
#'         this function.
#' }
#'
#' @param pts               Numeric matrix (n, 4) [x, y, z, stemcls].
#' @param k_neighbors       kNN for segment connectivity graph (default 6).
#' @param max_graph_distance Maximum Dijkstra path length in graph edge units
#'   (default 40; effectively unlimited for well-connected trees).
#' @param max_conn_dist     3D radius (m) used for point-level adjacency
#'   detection between segments (default 0.03 m).
#' @param occlusion_cutoff  Segments farther than this (m) from any adjacent
#'   segment are treated as occluded and edges are dropped (default 0.4 m).
#' @param min_pts_clean     Minimum points in a branch tip segment to keep
#'   that branch (default 5).
#' @param K_stem            kNN for stem cut-pursuit over-segmentation (default
#'   20; fewer neighbours → fewer, larger stem segments).
#' @param reg_stem          Cut-pursuit regularisation for stems (default 5.0;
#'   higher = fewer segments; lower = more fine-grained).
#' @param K_branch          kNN for branch cut-pursuit (default 3).
#' @param reg_branch        Cut-pursuit regularisation for branches (default
#'   0.01; very low → many small over-segments for accurate radius fitting).
#' @param min_radius_m      Minimum cylinder radius enforced after fitting
#'   (default 0.04 m = 4 cm; prevents degenerate cylinders).
#' @param threads           OpenMP threads for cut-pursuit (default 1).
#' @param verbose           Print step-level progress.
#'
#' @return Named list:
#' \describe{
#'   \item{tree}{List of integer vectors. Each vector is a branch path given
#'     as 1-based row indices into \code{segs_agg} (root → tip).}
#'   \item{segs_agg}{data.table: one row per over-segment with centroid XYZ,
#'     stem_label, point count, and 0-based seg label.}
#'   \item{segs_labels}{Integer vector (length n): per-point branch ID
#'     (1 = main stem, 2+ = branches, 0 = unassigned).}
#'   \item{tree_centroid_radius}{List of lists. Each inner list has one
#'     \code{numeric(4)} per skeleton node: [x, y, z, radius_m].}
#'   \item{init_segs}{Integer vector (length n): 0-based over-segmentation
#'     labels (raw cut-pursuit output).}
#' }
#' @export
apply_qsm <- function(pts,
                       k_neighbors          = 6L,
                       max_graph_distance   = 40,
                       max_conn_dist        = 0.03,
                       occlusion_cutoff     = 0.4,
                       min_pts_clean        = 5L,
                       K_stem               = 20L,
                       reg_stem             = 5.0,
                       K_branch             = 3L,
                       reg_branch           = 0.01,
                       min_radius_m         = 0.04,
                       threads              = 1L,
                       verbose              = TRUE) {
  if (!is.matrix(pts) || ncol(pts) < 4L)
    stop("pts must be a numeric matrix (n, 4+): [x, y, z, stemcls, ...]")
  if (!requireNamespace("RANN",   quietly = TRUE)) stop("RANN required.")
  if (!requireNamespace("igraph", quietly = TRUE)) stop("igraph required.")

  t0 <- Sys.time()

  if (verbose) message("[QSM] 1/5: init segmentation (cut-pursuit)...")
  init_segs <- .qsm_init_segmentation(pts,
                                       K_stem     = K_stem,
                                       reg_stem   = reg_stem,
                                       K_branch   = K_branch,
                                       reg_branch = reg_branch,
                                       threads    = threads)
  segs_agg <- .qsm_seg_centroids(pts, init_segs)
  n_segs   <- nrow(segs_agg)
  if (verbose) message(sprintf("[QSM]   %d over-segments from %d pts.", n_segs, nrow(pts)))

  if (verbose) message("[QSM] 2/5: find main stem path...")
  stem_idx <- .qsm_find_stems(segs_agg)
  if (length(stem_idx) == 0L) {
    warning("[QSM] No stem segments found -- using all segments as stem.")
    stem_idx <- seq_len(nrow(segs_agg))
  }
  if (verbose) message(sprintf("[QSM]   %d stem segments.", length(stem_idx)))

  if (verbose) message("[QSM] 3/5: build segment adjacency graph...")
  adj_pairs <- .qsm_seg_adjacency(pts, init_segs,
                                   k = k_neighbors,
                                   max_distance = max_conn_dist)
  graph <- .qsm_segment_graph(segs_agg, adj_pairs,
                               k_nn             = k_neighbors,
                               occlusion_cutoff = occlusion_cutoff)

  # Two-pass (mirrors updatePointwiseClusterDistance in Python):
  # first pass builds full skeleton; second pass catches poorly-connected
  # branches by restarting from all confirmed skeleton nodes.
  if (verbose) message("[QSM] 4/5: trace branch skeleton (two-pass)...")
  res1 <- .qsm_recreate_tree_path(stem_idx, graph, n_segs,
                                   max_graph_distance = max_graph_distance)
  # Second pass: all currently-labelled nodes become "reachable anchors"
  anchor_idx <- which(res1$segs_labels > 0L)
  res2 <- .qsm_recreate_tree_path(anchor_idx, graph, n_segs,
                                   max_graph_distance = max_graph_distance,
                                   init_seg_id        = res1$n_branches + 1L,
                                   segs_labels        = res1$segs_labels)
  tree        <- c(res1$tree, res2$tree[-1L])  # avoid duplicating stem path
  segs_labels <- res2$segs_labels

  clean       <- .qsm_clean_tree(tree, segs_agg$count, segs_labels,
                                  min_pts = min_pts_clean)
  tree        <- clean$tree
  segs_labels <- clean$segs_labels

  # Scatter per-segment labels back to per-point
  # segs_agg is ordered: row i+1 = seg label i  → segs_labels[init_segs+1]
  point_labels <- segs_labels[init_segs + 1L]

  if (verbose) message("[QSM] 5/5: fit cylinder radii...")
  tcr <- .qsm_calculate_radius(pts, tree, segs_agg, init_segs,
                                min_r = min_radius_m, verbose = verbose)

  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  n_br    <- length(tree)
  if (verbose)
    message(sprintf("[QSM] Done: %d branches | %.1f s.", n_br, elapsed))

  list(tree                 = tree,
       segs_agg             = segs_agg,
       segs_labels          = point_labels,
       tree_centroid_radius = tcr,
       init_segs            = init_segs)
}

# ---------------------------------------------------------------------------
# PUBLIC: qsm_to_cylinder_table
# ---------------------------------------------------------------------------

#' Convert apply_qsm() output to a flat cylinder table.
#'
#' Each row represents one cylinder (segment between two adjacent skeleton
#' nodes).  The columns follow the SimpleForest CSV convention so the table
#' can be imported into aRchi with
#' \code{aRchi::add_QSM(method = "SimpleForest")}.
#'
#' @param qsm_result  List returned by \code{\link{apply_qsm}}.
#' @param tree_id     Integer tree ID to stamp on every row (default 1L).
#' @param xyz_offset  Numeric vector of length 3.  Added back to all XYZ
#'   coordinates (use when \code{pts} was centred before calling
#'   \code{apply_qsm}).  Default \code{c(0, 0, 0)}.
#'
#' @return data.frame with columns \code{treeID}, \code{branchID},
#'   \code{parentBranchID}, \code{branchOrder}, \code{startX}, \code{startY},
#'   \code{startZ}, \code{endX}, \code{endY}, \code{endZ}, \code{radius},
#'   \code{length}.
#' @export
qsm_to_cylinder_table <- function(qsm_result, tree_id = 1L,
                                   xyz_offset = c(0, 0, 0)) {
  tree <- qsm_result$tree
  tcr  <- qsm_result$tree_centroid_radius
  rows <- vector("list", length(tree))
  for (bi in seq_along(tree)) {
    nodes <- tcr[[bi]]
    if (length(nodes) < 2L) next
    df <- data.frame(
      treeID         = tree_id,
      branchID       = bi,
      parentBranchID = if (bi == 1L) NA_integer_ else 1L,
      branchOrder    = bi - 1L,
      startX  = vapply(nodes[-length(nodes)], `[[`, numeric(1L), 1L) + xyz_offset[1L],
      startY  = vapply(nodes[-length(nodes)], `[[`, numeric(1L), 2L) + xyz_offset[2L],
      startZ  = vapply(nodes[-length(nodes)], `[[`, numeric(1L), 3L) + xyz_offset[3L],
      endX    = vapply(nodes[-1L],            `[[`, numeric(1L), 1L) + xyz_offset[1L],
      endY    = vapply(nodes[-1L],            `[[`, numeric(1L), 2L) + xyz_offset[2L],
      endZ    = vapply(nodes[-1L],            `[[`, numeric(1L), 3L) + xyz_offset[3L],
      radius  = (vapply(nodes[-length(nodes)], `[[`, numeric(1L), 4L) +
                 vapply(nodes[-1L],            `[[`, numeric(1L), 4L)) / 2,
      stringsAsFactors = FALSE)
    dx <- df$endX - df$startX; dy <- df$endY - df$startY; dz <- df$endZ - df$startZ
    df$length <- pmax(sqrt(dx^2 + dy^2 + dz^2), 1e-4)
    rows[[bi]] <- df
  }
  rows <- Filter(Negate(is.null), rows)
  if (length(rows) == 0L) return(data.frame())
  do.call(rbind, rows)
}
