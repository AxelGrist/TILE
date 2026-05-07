# treeisonet_pipeline.R
#
# R port of the NRCan TreeisoNet inference pipeline.
# Supports two sub-pipelines selected by the caller:
#
#   TLS / UAV (boreal, mixedwood):
#     1. treeisonet_run_treeloc()    – detection model → 3D tree-base locations
#     2. treeisonet_shortestpath3D() – kNN Dijkstra graph → per-point tree IDs
#
#   ALS (reclamation):
#     1. treeisonet_run_treeloc()    – same as above
#     2. treeisonet_run_treeoff()    – regression model → per-point (dx,dy) offsets
#     3. treeisonet_mergeshift()     – shift points → nearest treeloc → tree ID
#
# StemCls inference (step 0 for TLS/UAV) re-uses classify_wood() in classify.R
# because the stemcls model is a standard ESegformer3D segmentation head.
#
# the stemcls model is a standard ESegformer3D segmentation head.
#
# References:
#   Xi Z, Hopkinson C, Chasmer L (2018) Remote Sens 10:1215
#   Xi Z, Chasmer L, Hopkinson C (2023) Remote Sens 15:4778
#   https://github.com/NRCan/TreeAIBox (CC BY-NC 4.0)
#
# Dependencies:
#   torch (build-time), RANN (kNN), igraph (Dijkstra + components)

# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

# BFS connected-component labeling on a logical 2D matrix.
# Returns list(labels=integer_matrix, n_labels=integer).
.label_connected_2d <- function(mat) {
  nr <- nrow(mat); nc <- ncol(mat)
  labels     <- matrix(0L, nr, nc)
  next_label <- 1L
  neighbours <- rbind(c(-1L, 0L), c(1L, 0L), c(0L, -1L), c(0L, 1L),
                      c(-1L,-1L), c(-1L, 1L), c(1L,-1L), c(1L, 1L))
  for (r in seq_len(nr)) {
    for (cc in seq_len(nc)) {
      if (!mat[r, cc] || labels[r, cc] != 0L) next
      # BFS flood fill
      queue   <- matrix(c(r, cc), ncol = 2L)
      labels[r, cc] <- next_label
      head_i  <- 1L
      while (head_i <= nrow(queue)) {
        cur   <- queue[head_i, ]
        head_i <- head_i + 1L
        for (k in seq_len(nrow(neighbours))) {
          nr2 <- cur[1L] + neighbours[k, 1L]
          nc2 <- cur[2L] + neighbours[k, 2L]
          if (nr2 >= 1L && nr2 <= nr && nc2 >= 1L && nc2 <= nc &&
              mat[nr2, nc2] && labels[nr2, nc2] == 0L) {
            labels[nr2, nc2] <- next_label
            queue <- rbind(queue, c(nr2, nc2))
          }
        }
      }
      next_label <- next_label + 1L
    }
  }
  list(labels = labels, n_labels = next_label - 1L)
}

# Distance-based 2D non-maximum suppression.
# candidates: matrix (n, 3) [x, y, z]; nms_thresh_xy in metres.
.nms3d <- function(candidates, nms_thresh_xy = 0.3) {
  if (nrow(candidates) <= 1L) return(candidates)
  ord  <- order(candidates[, 3L])
  cand <- candidates[ord, , drop = FALSE]
  keep <- rep(TRUE, nrow(cand))
  for (i in seq_len(nrow(cand) - 1L)) {
    if (!keep[i]) next
    dists_xy <- sqrt((cand[(i+1L):nrow(cand), 1L] - cand[i, 1L])^2 +
                     (cand[(i+1L):nrow(cand), 2L] - cand[i, 2L])^2)
    suppress <- which(dists_xy < nms_thresh_xy)
    if (length(suppress)) keep[(i + suppress)] <- FALSE
  }
  cand[keep, , drop = FALSE]
}

# Merge per-block 2D prediction patches into a global XY image.
# pred_patches: list of logical matrices (X_vox, Y_vox) per block.
# sp_mins_list: list of numeric(3) XYZ block origins.
.merge_patches2d <- function(pred_patches, sp_mins_list,
                              pcd_min, pcd_max, res_xy) {
  img_nx <- as.integer(floor((pcd_max[1L] - pcd_min[1L]) / res_xy[1L])) + 2L
  img_ny <- as.integer(floor((pcd_max[2L] - pcd_min[2L]) / res_xy[2L])) + 2L
  img    <- matrix(FALSE, img_nx, img_ny)
  for (k in seq_along(pred_patches)) {
    p   <- pred_patches[[k]]
    if (is.null(p) || length(p) == 0L) next
    ox  <- as.integer(floor((sp_mins_list[[k]][1L] - pcd_min[1L]) / res_xy[1L]))
    oy  <- as.integer(floor((sp_mins_list[[k]][2L] - pcd_min[2L]) / res_xy[2L]))
    px  <- nrow(p); py <- ncol(p)
    xi  <- seq_len(px) + ox
    yi  <- seq_len(py) + oy
    xi  <- xi[xi >= 1L & xi <= img_nx]
    yi  <- yi[yi >= 1L & yi <= img_ny]
    img[xi, yi] <- img[xi, yi] | p[seq_along(xi), seq_along(yi)]
  }
  img
}

# --------------------------------------------------------------------------
# treeisonet_remove_small_clusters
# --------------------------------------------------------------------------

#' Remove small isolated stem clusters after StemCls inference.
#'
#' Builds a 3D proximity graph over stem points and removes connected
#' components with fewer than \code{min_points} members.  Mirrors the
#' "Remove small clusters" checkbox in the TreeAIBox GUI.
#'
#' @param stemcls     integer vector (length n): classification labels where
#'   values >= 2 are treated as stem.
#' @param xyz         numeric matrix (n, 3) of XYZ coordinates.
#' @param max_gap_m   maximum 3D distance (m) to consider two stem points
#'   connected (GUI "Max gap").
#' @param min_points  minimum cluster size to retain as stem
#'   (GUI "Min points").
#' @return integer vector of the same length as \code{stemcls} with small
#'   clusters reclassified to 1 (foliage).
#' @export
treeisonet_remove_small_clusters <- function(stemcls, xyz,
                                             max_gap_m  = 3.0,
                                             min_points = 100L) {
  if (!requireNamespace("RANN",   quietly = TRUE))
    stop("treeisonet_remove_small_clusters() requires RANN.")
  if (!requireNamespace("igraph", quietly = TRUE))
    stop("treeisonet_remove_small_clusters() requires igraph.")

  stem_idx <- which(stemcls >= 2L)
  if (length(stem_idx) == 0L) return(stemcls)

  # Trivial: all stem points are already below the threshold
  if (length(stem_idx) < min_points) {
    stemcls[stem_idx] <- 1L
    return(stemcls)
  }

  xyz_stem <- xyz[stem_idx, 1:3L, drop = FALSE]
  n_stem   <- nrow(xyz_stem)

  # kNN adjacency graph capped at max_gap_m
  k_nn    <- min(10L, n_stem - 1L)
  nn      <- RANN::nn2(xyz_stem, xyz_stem, k = k_nn + 1L)
  from_v  <- rep(seq_len(n_stem), each = k_nn)
  to_v    <- as.vector(t(nn$nn.idx[, -1L, drop = FALSE]))
  dist_v  <- as.vector(t(nn$nn.dists[, -1L, drop = FALSE]))
  valid   <- dist_v <= max_gap_m & from_v != to_v
  from_v  <- from_v[valid]
  to_v    <- to_v[valid]

  if (length(from_v) == 0L) {
    stemcls[stem_idx] <- 1L
    return(stemcls)
  }

  g     <- igraph::graph_from_data_frame(
    data.frame(from = from_v, to = to_v),
    directed = FALSE,
    vertices = data.frame(name = seq_len(n_stem))
  )
  comps <- igraph::components(g)
  small <- which(comps$csize < min_points)
  if (length(small) > 0L)
    stemcls[stem_idx[comps$membership %in% small]] <- 1L

  stemcls
}

# --------------------------------------------------------------------------
# treeisonet_run_treeloc
# --------------------------------------------------------------------------

#' Detect individual tree base locations with the TreeisoNet treeloc model.
#'
#' Runs the Detection-stem ESegformer3D model on stem-only columns, builds a
#' 2D confidence map, finds peaks, and returns 3D base XYZ coordinates.
#'
#' @param xyz         numeric matrix (n, 3) of XYZ coordinates (stem-only or full cloud).
#' @param model_bundle list returned by \code{load_treeaibox_model()}.
#' @param cutoff_thresh fraction of per-tile stem height to use (TLS/UAV path, GUI "Cutoff height ratio", default 0.3).
#' @param cut_grid_res  tile width (m) for height normalisation before cutoff filter.
#' @param conf_thresh   minimum per-point confidence for peak map (ALS path, GUI "Confidence", default 0.3).
#' @param nms_thresh_xy NMS factor: TLS/UAV fixed-radius (m); ALS applied as (r_i+r_j)*nms_thresh_xy (GUI "NMS cutoff", default 0.5).
#' @param base_radius_m radius (m) for snapping each peak to the nearest real point (GUI "Minimum radius", default 0.2).
#' @param treeloc_max_gap max 3D gap (m) for grouping nearby confidence peaks in ALS \code{postPeakExtraction} kNN graph (GUI "Max gap", default 0.3).
#' @param treeloc_K     kNN k for ALS peak-grouping graph in \code{postPeakExtraction} (hardcoded 5 in GUI, default 5L).
#' @param vox_override  named list with \code{xy} and \code{z} (metres) to override model voxel resolution; NA/NULL uses model config.
#' @param verbose       print progress.
#' @return matrix (n_trees, 3) of detected base XYZ locations, or empty matrix.
#' @export
treeisonet_run_treeloc <- function(xyz,
                                   model_bundle,
                                   cutoff_thresh   = 0.3,
                                   cut_grid_res    = 5.0,
                                   conf_thresh     = 0.3,
                                   nms_thresh_xy   = 0.5,
                                   base_radius_m   = 0.2,
                                   treeloc_max_gap = 0.3,
                                   treeloc_K       = 5L,
                                   vox_override    = NULL,
                                   verbose         = FALSE) {
  if (!requireNamespace("RANN", quietly = TRUE))
    stop("treeisonet_run_treeloc() requires the RANN package.")

  model  <- model_bundle$model
  device <- model_bundle$device
  cfg    <- model_bundle$config$model
  nbmat_sz  <- as.integer(cfg$voxel_number_in_block)     # c(X, Y, Z) voxels
  min_res   <- as.numeric(cfg$voxel_resolution_in_meter) # c(rx, ry, rz) m
  if (!is.null(vox_override$xy) && !is.na(vox_override$xy))
    min_res[1:2L] <- as.numeric(vox_override$xy)
  if (!is.null(vox_override$z) && !is.na(vox_override$z))
    min_res[3L] <- as.numeric(vox_override$z)
  nb_tsz    <- prod(nbmat_sz)

  # NOTE: treeloc_max_gap and treeloc_K are ALS-path parameters used in
  # postPeakExtraction() (see treeLoc.py). They are not yet applied here
  # because this function currently implements the TLS/UAV detection path
  # (2D heatmap → connected components). When the ALS per-point regression
  # path is added, these two values replace the kNN gap and k arguments to
  # the postPeakExtraction equivalent below.

  # 1 ---- Filter to lower portion of stems per 5 m tile --------------------
  pcd_xy <- xyz[, 1:2L, drop = FALSE]
  pcd_min <- apply(xyz, 2L, min)

  tile_ij <- floor((pcd_xy - rep(pcd_min[1:2L], each = nrow(xyz))) /
                   cut_grid_res)
  tile_key <- paste(tile_ij[, 1L], tile_ij[, 2L], sep = "_")
  tile_groups <- split(seq_len(nrow(xyz)), tile_key)

  filter_idx <- unlist(lapply(tile_groups, function(idx) {
    zvals    <- xyz[idx, 3L]
    z_min    <- min(zvals)
    z_range  <- max(zvals) - z_min
    idx[zvals - z_min <= z_range * cutoff_thresh]
  }), use.names = FALSE)

  if (length(filter_idx) == 0L) {
    if (verbose) message("treeloc: no stem points after height filter.")
    return(matrix(NA_real_, 0L, 3L))
  }
  xyz_stem  <- xyz[filter_idx, , drop = FALSE]
  stem_min  <- apply(xyz_stem, 2L, min)

  if (verbose)
    message(sprintf("treeloc: %d stem points after height filter.", nrow(xyz_stem)))

  # 2 ---- Column-by-column block processing --------------------------------
  block_ij <- floor(
    (xyz_stem[, 1:2L, drop = FALSE] - rep(stem_min[1:2L], each = nrow(xyz_stem))) /
    (min_res[1:2L] * nbmat_sz[1:2L])
  )
  block_key    <- paste(block_ij[, 1L], block_ij[, 2L], sep = "_")
  block_groups <- split(seq_len(nrow(xyz_stem)), block_key)

  pred_patches <- vector("list", length(block_groups))
  sp_mins_list <- vector("list", length(block_groups))

  nblk_treeloc <- length(block_groups)
  .pb_treeloc <- if (!verbose && nblk_treeloc > 1L) {
    function(i) {
      filled <- round(30L * i / nblk_treeloc)
      cat(sprintf("\r  [%s] %3d%%  (%d/%d blocks)",
                  paste0(strrep("=", filled), strrep(" ", 30L - filled)),
                  round(100L * i / nblk_treeloc), i, nblk_treeloc),
          file = stderr())
      if (i == nblk_treeloc) cat("\n", file = stderr())
    }
  } else function(i) invisible(NULL)

  model$eval()
  for (k in seq_along(block_groups)) {
    .pb_treeloc(k)
    idx       <- block_groups[[k]]
    pts_block <- xyz_stem[idx, , drop = FALSE]
    sp_min    <- apply(pts_block, 2L, min)

    # Voxelize: compute ijk in (X, Y, Z) order to match ravel_multi_index
    ijk <- floor((pts_block - rep(sp_min, each = nrow(pts_block))) /
                 rep(min_res, each = nrow(pts_block)))
    valid <- apply(ijk < rep(nbmat_sz, each = nrow(ijk)) &
                   ijk >= 0L, 1L, all)
    ijk   <- ijk[valid, , drop = FALSE]

    # Row-major flat index: X*Y*Z + Y*Z + Z  (nbmat_sz = c(X,Y,Z))
    flat_idx <- as.integer(ijk[, 1L]) * nbmat_sz[2L] * nbmat_sz[3L] +
                as.integer(ijk[, 2L]) * nbmat_sz[3L] +
                as.integer(ijk[, 3L])
    # Handle duplicates — keep unique voxels
    flat_idx <- unique(flat_idx)

    # Build input tensor (B=1, C=1, X, Y, Z) → swapaxes → (B, C, Z, Y, X)
    x_flat <- torch_zeros(nb_tsz, 1L)
    if (length(flat_idx) > 0L)
      x_flat[flat_idx + 1L, 1L] <- 1.0   # 1-indexed R torch
    x <- x_flat$reshape(c(1L, nbmat_sz[1L], nbmat_sz[2L], nbmat_sz[3L], 1L))$
           permute(c(1L, 5L, 2L, 3L, 4L))$   # (B,C,X,Y,Z)
           transpose(3L, 5L)$                  # (B,C,Z,Y,X)
           to(device = device)

    # Run detection model → h: (B, num_classes, Y_out, X_out)
    with_no_grad({
      h <- model(x)
    })
    # swapaxes(2, -1) for 4-D: swap dim3 (Y) and dim4 (X) → (B, nc, X, Y)
    h_4d <- h$transpose(3L, 4L)
    # Class 1 (0-indexed) = tree class (1-indexed in R torch = 2nd channel)
    nc   <- h_4d$size(2L)
    if (nc >= 2L) {
      probs <- nnf_softmax(h_4d, dim = 2L)[1L, 2L, , ]  # (X, Y)
    } else {
      probs <- torch_sigmoid(h_4d[1L, 1L, , ])
    }
    pred_patches[[k]] <- as.array(probs$cpu()) > conf_thresh
    sp_mins_list[[k]] <- sp_min
  }

  if (verbose) message(sprintf("treeloc: processed %d blocks.", length(block_groups)))

  # 3 ---- Merge patches into global 2D image --------------------------------
  img <- .merge_patches2d(pred_patches, sp_mins_list,
                           pcd_min = stem_min,
                           pcd_max = apply(xyz_stem, 2L, max),
                           res_xy  = min_res[1:2L])

  if (!any(img)) {
    if (verbose) message("treeloc: empty confidence image — no trees detected.")
    return(matrix(NA_real_, 0L, 3L))
  }

  # 4 ---- Peak finder: connected components in 2D image --------------------
  labeled <- .label_connected_2d(img)
  if (labeled$n_labels == 0L) return(matrix(NA_real_, 0L, 3L))

  # Centroid of each component → XY world coords
  centroids_xy <- do.call(rbind, lapply(seq_len(labeled$n_labels), function(lbl) {
    pix  <- which(labeled$labels == lbl, arr.ind = TRUE)
    cent <- colMeans(pix) - 1L   # 0-indexed
    stem_min[1:2L] + cent * min_res[1:2L]
  }))

  # 5 ---- For each peak: find lowest-Z point within base_radius_m ----------
  nn_res <- RANN::nn2(xyz_stem[, 1:2L, drop = FALSE],
                      centroids_xy, k = 1L)
  base_locs <- do.call(rbind, lapply(seq_len(nrow(centroids_xy)), function(k) {
    cx <- centroids_xy[k, 1L]; cy <- centroids_xy[k, 2L]
    d2 <- sqrt((xyz_stem[, 1L] - cx)^2 + (xyz_stem[, 2L] - cy)^2)
    nearby <- which(d2 < base_radius_m)
    if (length(nearby) == 0L) nearby <- nn_res$nn.idx[k, 1L]
    xyz_stem[nearby[which.min(xyz_stem[nearby, 3L])], , drop = FALSE]
  }))

  # 6 ---- Non-maximum suppression ------------------------------------------
  base_locs <- .nms3d(base_locs, nms_thresh_xy = nms_thresh_xy)

  if (verbose)
    message(sprintf("treeloc: %d tree bases detected.", nrow(base_locs)))

  base_locs
}

# --------------------------------------------------------------------------
# treeisonet_shortestpath3D
# --------------------------------------------------------------------------

#' Assign points to individual trees via Dijkstra shortest-path graph.
#'
#' R port of \code{stemCluster.shortestpath3D()} from TreeAIBox. Takes the
#' voxel-decimated stem point cloud, builds a kNN adjacency graph, finds
#' connected components, locates which tree bases fall in which component,
#' splits multi-base components with k-means, then runs Dijkstra from each
#' base node and assigns each point to its nearest base.
#'
#' @param xyz             numeric matrix (n, 3) of ALL points (not just stem).
#' @param stemcls         integer vector length n: 1=foliage, 2=branch, 3=stem
#'   (values >= 2 are treated as "stem" for graph construction).
#' @param base_locs       matrix (b, 3) of tree-base XYZ from
#'   \code{treeisonet_run_treeloc()}.
#' @param min_res         voxel edge length for decimation (m).
#' @param max_isolated_dist  maximum 3D edge length in kNN graph (m).
#' @param k_graph         kNN k for point-level graph edges.
#' @param k_node          kNN k for component-centroid node graph.
#' @param verbose         print progress.
#' @return integer vector length n: tree ID (1-indexed) for each point;
#'   0 = unassigned (isolated / no path to any base).
#' @export
treeisonet_shortestpath3D <- function(xyz,
                                      stemcls,
                                      base_locs,
                                      min_res              = 0.06,
                                      max_isolated_dist    = 0.3,
                                      k_graph              = 10L,
                                      k_node               = 20L,
                                      verbose              = FALSE) {
  if (!requireNamespace("igraph", quietly = TRUE))
    stop("treeisonet_shortestpath3D() requires igraph. ",
         "Install with: install.packages('igraph')")
  if (!requireNamespace("RANN", quietly = TRUE))
    stop("treeisonet_shortestpath3D() requires RANN.")

  out <- rep(0L, nrow(xyz))

  # --- Guard: need bases ------------------------------------------------
  if (is.null(base_locs) || nrow(base_locs) == 0L) {
    warning("treeisonet_shortestpath3D: no base locations provided.")
    return(out)
  }

  # 1. Filter to stem points -------------------------------------------
  stem_idx <- which(stemcls >= 2L)
  if (length(stem_idx) == 0L) {
    warning("treeisonet_shortestpath3D: no stem-class points found.")
    return(out)
  }
  xyz_stem <- xyz[stem_idx, 1:3L, drop = FALSE]

  # 2. Voxel decimation ------------------------------------------------
  ijk_key <- paste(floor(xyz_stem[, 1L] / min_res),
                   floor(xyz_stem[, 2L] / min_res),
                   floor(xyz_stem[, 3L] / min_res), sep = "_")
  dec_u_mask   <- !duplicated(ijk_key)
  dec_uidx     <- which(dec_u_mask)            # stem indices of decimated pts
  dec_inverse  <- match(ijk_key, ijk_key[dec_u_mask])  # stem_pt → dec_pt index
  xyz_dec      <- xyz_stem[dec_uidx, , drop = FALSE]
  n_dec        <- nrow(xyz_dec)
  if (verbose) message(sprintf("shortestpath3D: %d stem pts → %d decimated.", nrow(xyz_stem), n_dec))

  # --- Trivial cases ---
  if (n_dec == 1L) {
    out[stem_idx] <- 1L; return(out)
  }

  # 3. kNN sparse point graph -------------------------------------------
  k_g <- min(k_graph, n_dec - 1L)
  nn  <- RANN::nn2(xyz_dec, xyz_dec, k = k_g + 1L)
  nn_idx  <- nn$nn.idx[, -1L, drop = FALSE]   # exclude self
  nn_dist <- nn$nn.dists[, -1L, drop = FALSE]

  edge_from <- rep(seq_len(n_dec), each = k_g)
  edge_to   <- as.vector(t(nn_idx))
  edge_w    <- as.vector(t(nn_dist))
  # Upstream Python uses min_res * 3 as the point-level graph edge cutoff
  # (max_isolated_dist applies only to the node-level graph, see stemCluster.py)
  valid     <- edge_w < min_res * 3 & edge_from != edge_to
  edge_from <- edge_from[valid]; edge_to <- edge_to[valid]; edge_w <- edge_w[valid]

  if (length(edge_from) == 0L) {
    # All isolated: nearest-base assignment
    nn_b <- RANN::nn2(base_locs[, 1:3L, drop = FALSE],
                      xyz[, 1:3L, drop = FALSE], k = 1L)
    return(as.integer(nn_b$nn.idx[, 1L]))
  }

  g_pts <- igraph::graph_from_data_frame(
    data.frame(from = edge_from, to = edge_to, weight = edge_w),
    directed = FALSE, vertices = data.frame(name = seq_len(n_dec))
  )

  # 4. Connected components -----------------------------------------------
  comps       <- igraph::components(g_pts)
  comp_labels <- comps$membership   # length n_dec; 1-indexed component IDs

  # 5. Find nearest decimated node to each base ---------------------------
  nn_b2d <- RANN::nn2(xyz_dec, base_locs[, 1:3L, drop = FALSE], k = 1L)
  base_node_ids <- as.integer(nn_b2d$nn.idx[, 1L])  # dec-node index per base
  n_bases       <- nrow(base_locs)

  # 6. K-means split for components with >1 base --------------------------
  base_comp     <- comp_labels[base_node_ids]  # component of each base
  comp_n_bases  <- tapply(seq_len(n_bases), base_comp, length)
  multi_comps   <- as.integer(names(comp_n_bases)[comp_n_bases > 1L])

  if (length(multi_comps) > 0L) {
    max_lbl <- max(comp_labels)
    for (cid in multi_comps) {
      comp_pts_idx <- which(comp_labels == cid)
      bases_in_c   <- which(base_comp == cid)
      nb           <- length(bases_in_c)
      if (nb <= 1L || length(comp_pts_idx) < nb) next
      km <- tryCatch(
        stats::kmeans(xyz_dec[comp_pts_idx, , drop = FALSE],
                      centers = nb, iter.max = 100L, nstart = 3L),
        error = function(e) NULL)
      if (is.null(km)) next
      for (kk in seq_len(nb)) {
        if (kk == 1L) next     # keep cid for first cluster
        sub_idx <- comp_pts_idx[km$cluster == kk]
        max_lbl <- max_lbl + 1L
        comp_labels[sub_idx] <- max_lbl
      }
    }
    # Re-derive after split
    base_comp <- comp_labels[base_node_ids]
  }

  # 7. Build node graph (component centroids) -----------------------------
  unique_comps <- sort(unique(comp_labels))
  n_comps      <- length(unique_comps)
  comp_map     <- setNames(seq_len(n_comps), as.character(unique_comps))

  centroids <- do.call(rbind, lapply(unique_comps, function(cid) {
    colMeans(xyz_dec[comp_labels == cid, , drop = FALSE])
  }))

  if (n_comps > 1L) {
    k_n <- min(k_node, n_comps - 1L)
    nn_c <- RANN::nn2(centroids, centroids, k = k_n + 1L)
    ne_from <- rep(seq_len(n_comps), each = k_n)
    ne_to   <- as.vector(t(nn_c$nn.idx[, -1L, drop = FALSE]))
    ne_w    <- as.vector(t(nn_c$nn.dists[, -1L, drop = FALSE]))
    valid_e <- ne_w < max_isolated_dist * 2L & ne_from != ne_to
    ne_from <- ne_from[valid_e]; ne_to <- ne_to[valid_e]; ne_w <- ne_w[valid_e]
  } else {
    ne_from <- integer(0L); ne_to <- integer(0L); ne_w <- numeric(0L)
  }

  if (length(ne_from) == 0L) {
    # No inter-component edges: assign by nearest base centroid
    nn_c2b <- RANN::nn2(base_locs[, 1:3L, drop = FALSE], centroids, k = 1L)
    comp_to_tree <- as.integer(nn_c2b$nn.idx[, 1L])
    tree_ids_stem <- comp_to_tree[comp_map[as.character(comp_labels)][dec_inverse]]
    out[stem_idx] <- as.integer(tree_ids_stem)
    return(out)
  }

  g_node <- igraph::graph_from_data_frame(
    data.frame(from = ne_from, to = ne_to, weight = ne_w),
    directed = FALSE, vertices = data.frame(name = seq_len(n_comps))
  )

  # 8. Dijkstra from base nodes -------------------------------------------
  base_comp_node_idx <- comp_map[as.character(base_comp)]   # node in g_node
  target_node_idxs   <- unique(base_comp_node_idx)          # unique g_node targets

  dist_mat <- igraph::distances(
    g_node,
    v  = igraph::V(g_node)[target_node_idxs],
    to = igraph::V(g_node),
    weights = igraph::E(g_node)$weight
  )   # (n_targets, n_comps) — distances from each target to every node

  dist_mat <- t(dist_mat)    # (n_comps, n_targets)

  # 9. Assign each component node to the nearest target base ---------------
  best_col       <- apply(dist_mat, 1L, which.min)       # length n_comps
  min_dists      <- dist_mat[cbind(seq_len(n_comps), best_col)]

  # Map target column → original base (tree) index
  target_to_base <- sapply(seq_along(target_node_idxs), function(ti) {
    matches <- which(base_comp_node_idx == target_node_idxs[ti])
    if (length(matches)) matches[1L] else 0L
  })
  comp_tree_ids         <- target_to_base[best_col]
  comp_tree_ids[is.infinite(min_dists)] <- 0L   # unreachable

  # 10. Map back: comp label → tree id → all stem pts → all pts ------------
  node_idx_for_comp <- comp_map[as.character(comp_labels)]   # length n_dec
  tree_ids_dec      <- comp_tree_ids[node_idx_for_comp]      # length n_dec
  tree_ids_stem     <- tree_ids_dec[dec_inverse]             # length = |stem_idx|
  out[stem_idx]     <- as.integer(tree_ids_stem)
  out
}

# --------------------------------------------------------------------------
# treeisonet_run_treeoff  (ALS / reclamation scene)
# --------------------------------------------------------------------------

#' Predict per-point XY offset to the nearest tree stem using the TreeOff
#' regression model.
#'
#' R port of \code{treeOff.treeOff()} from TreeAIBox.  The model takes a
#' 2-channel voxel block (channel 0 = point occupancy; channel 1 = treeloc
#' indicator broadcast along Z) and outputs per-voxel (dx,dy) offsets in
#' voxel units.  The offsets are scaled by \code{min_res[1:2]} to metres
#' before being returned.
#'
#' @param xyz            numeric matrix (n, 3) of all point XYZ.
#' @param treelocs       matrix (b, 3) of tree-base XYZ from
#'   \code{treeisonet_run_treeloc()}.
#' @param model_bundle   list from \code{load_treeaibox_model()} for a
#'   \code{treeoff} or \code{crownoff} model (\code{head_type="regression"}).
#' @param verbose        print progress.
#' @return numeric matrix (n, 2): per-point (dx,dy) offset in metres.
#'   Points that fell outside every block have offset (0,0).
#' @export
treeisonet_run_treeoff <- function(xyz, treelocs, model_bundle,
                                   vox_override = NULL, verbose = FALSE) {
  model    <- model_bundle$model
  device   <- model_bundle$device
  cfg      <- model_bundle$config$model
  nbmat_sz <- as.integer(cfg$voxel_number_in_block)   # c(X, Y, Z)
  min_res  <- as.numeric(cfg$voxel_resolution_in_meter)
  if (!is.null(vox_override$xy) && !is.na(vox_override$xy))
    min_res[1:2L] <- as.numeric(vox_override$xy)
  if (!is.null(vox_override$z) && !is.na(vox_override$z))
    min_res[3L] <- as.numeric(vox_override$z)
  nb_tsz   <- prod(nbmat_sz)
  nb_tsz2d <- nbmat_sz[1L] * nbmat_sz[2L]             # XY voxels per block

  pcd_min <- apply(xyz, 2L, min)

  # Append row index as tree-ID column (0-indexed → 1-indexed after +1 in
  # treeisonet_mergeshift)
  treelocs_id <- cbind(treelocs[, 1:3L, drop = FALSE],
                       seq_len(nrow(treelocs)) - 1L)  # 0-indexed tree id

  # Column-by-column block grouping (same as treeloc)
  block_ij <- floor(
    (xyz[, 1:2L, drop = FALSE] - rep(pcd_min[1:2L], each = nrow(xyz))) /
    (min_res[1:2L] * nbmat_sz[1:2L])
  )
  block_key    <- paste(block_ij[, 1L], block_ij[, 2L], sep = "_")
  block_groups <- split(seq_len(nrow(xyz)), block_key)

  pcd_pred      <- matrix(0, nrow(xyz), 2L)   # per-point (dx_m, dy_m)
  pcd_within    <- logical(nrow(xyz))

  nblk_treeoff <- length(block_groups)
  .pb_treeoff <- if (!verbose && nblk_treeoff > 1L) {
    function(i) {
      filled <- round(30L * i / nblk_treeoff)
      cat(sprintf("\r  [%s] %3d%%  (%d/%d blocks)",
                  paste0(strrep("=", filled), strrep(" ", 30L - filled)),
                  round(100L * i / nblk_treeoff), i, nblk_treeoff),
          file = stderr())
      if (i == nblk_treeoff) cat("\n", file = stderr())
    }
  } else function(i) invisible(NULL)

  model$eval()
  for (k in seq_along(block_groups)) {
    .pb_treeoff(k)
    idx       <- block_groups[[k]]
    pts_block <- xyz[idx, 1:3L, drop = FALSE]
    sp_min    <- apply(pts_block, 2L, min)
    sp_max    <- apply(pts_block, 2L, max)

    # Treelocs in this block's XY footprint
    in_blk <- apply(
      (matrix(rep(sp_max[1:2L], nrow(treelocs_id)), nrow = nrow(treelocs_id),
              byrow = TRUE) - treelocs_id[, 1:2L]) > 0 &
      (treelocs_id[, 1:2L] - matrix(rep(sp_min[1:2L], nrow(treelocs_id)),
                                    nrow = nrow(treelocs_id), byrow = TRUE)) > 0,
      1L, all)
    treelocs_sp <- treelocs_id[in_blk, , drop = FALSE]

    # Voxelize block points → flat 3D indices
    ijk <- floor((pts_block - rep(sp_min, each = nrow(pts_block))) /
                 rep(min_res, each = nrow(pts_block)))
    valid <- apply(ijk < rep(nbmat_sz, each = nrow(ijk)) & ijk >= 0L, 1L, all)
    ijk_v <- ijk[valid, , drop = FALSE]
    idx_v <- idx[valid]

    flat3d <- as.integer(ijk_v[, 1L]) * nbmat_sz[2L] * nbmat_sz[3L] +
              as.integer(ijk_v[, 2L]) * nbmat_sz[3L] +
              as.integer(ijk_v[, 3L])

    unq3d      <- unique(flat3d)
    inv3d      <- match(flat3d, unq3d)

    # Channel 0: 3D occupancy
    x_occ <- torch_zeros(nb_tsz, 1L)
    if (length(unq3d) > 0L)
      x_occ[unq3d + 1L, 1L] <- 1.0
    x_occ <- x_occ$reshape(c(1L, nbmat_sz[1L], nbmat_sz[2L], nbmat_sz[3L], 1L))$
               permute(c(1L, 5L, 2L, 3L, 4L))$    # (B,C,X,Y,Z)
               transpose(3L, 5L)$                   # (B,C,Z,Y,X)
               to(device = device)

    # Channel 1: treeloc 2D indicator broadcast to Z
    # treeloc XY voxel coords within block
    x_loc_2d <- torch_zeros(nb_tsz2d, 1L)
    if (nrow(treelocs_sp) > 0L) {
      tloc_ij  <- floor((treelocs_sp[, 1:2L, drop = FALSE] -
                         rep(sp_min[1:2L], each = nrow(treelocs_sp))) /
                        min_res[1:2L])
      tloc_ok  <- apply(tloc_ij < rep(nbmat_sz[1:2L], each = nrow(tloc_ij)) &
                        tloc_ij >= 0L, 1L, all)
      tloc_ij  <- tloc_ij[tloc_ok, , drop = FALSE]
      if (nrow(tloc_ij) > 0L) {
        flat2d <- as.integer(tloc_ij[, 1L]) * nbmat_sz[2L] +
                  as.integer(tloc_ij[, 2L])
        flat2d <- unique(flat2d)
        x_loc_2d[flat2d + 1L, 1L] <- 1.0
      }
    }
    x_loc_3d <- x_loc_2d$reshape(c(1L, nbmat_sz[1L], nbmat_sz[2L], 1L))$
                  permute(c(1L, 4L, 2L, 3L))$          # (B,1,X,Y)
                  unsqueeze(3L)$                         # (B,1,1,X,Y) → unsqueeze Z
                  expand(c(1L, 1L, nbmat_sz[3L], nbmat_sz[1L], nbmat_sz[2L]))$
                  to(device = device)

    # Concatenate along channel dim → (B,2,Z,Y,X)
    x_in <- torch_cat(list(x_occ, x_loc_3d), dim = 2L)

    # Forward → h: (B, 2, Z_out, Y_out, X_out)  [regression head]
    with_no_grad({
      h <- model(x_in)
    })

    # Extract predictions at occupied voxels and map back to points
    # h shape: (B, 2, Z, Y, X) → need (nb_tsz, 2) layout
    # swapaxes(-1,2): (B,2,X,Y,Z) → permute(0,1,4,3,2) would also work
    h_flat <- h$transpose(3L, 5L)$           # (B,2,X,Y,Z)
               permute(c(1L, 3L, 4L, 5L, 2L))$  # (B,X,Y,Z,2)
               reshape(c(nb_tsz, 2L))            # (nb_tsz, 2)
    pred_at_pts <- as.matrix(h_flat[unq3d + 1L, ]$cpu())  # (n_unq, 2)

    pcd_pred[idx_v, ] <- pred_at_pts[inv3d, ]  # propagate to every point
    pcd_within[idx_v] <- TRUE

  }

  # Scale voxel-unit offsets to metres
  pcd_pred[, 1L] <- pcd_pred[, 1L] * min_res[1L]
  pcd_pred[, 2L] <- pcd_pred[, 2L] * min_res[2L]
  pcd_pred
}

# --------------------------------------------------------------------------
# treeisonet_mergeshift  (ALS / reclamation scene)
# --------------------------------------------------------------------------

#' Assign points to trees by shifting predicted offsets and nearest-treeloc lookup.
#'
#' R port of \code{treeOff.mergeshift()}.  Shifts each point's XY by its
#' predicted offset, then assigns it to the nearest tree-base in 2D.
#'
#' @param xyz         numeric matrix (n, 3) original XYZ.
#' @param offsets     numeric matrix (n, 2) per-point (dx_m, dy_m) from
#'   \code{treeisonet_run_treeoff()}.
#' @param treelocs    matrix (b, 3) of tree-base XYZ.
#' @return integer vector length n: tree ID (1-indexed); 0 = unassigned
#'   (only for points that were never in any block, i.e. offsets==0 AND
#'   treelocs list is empty).
#' @export
treeisonet_mergeshift <- function(xyz, offsets, treelocs) {
  if (!requireNamespace("RANN", quietly = TRUE))
    stop("treeisonet_mergeshift() requires the RANN package.")
  if (is.null(treelocs) || nrow(treelocs) == 0L)
    return(rep(0L, nrow(xyz)))

  xyz_shifted     <- xyz[, 1:2L, drop = FALSE]
  xyz_shifted[, 1L] <- xyz_shifted[, 1L] + offsets[, 1L]
  xyz_shifted[, 2L] <- xyz_shifted[, 2L] + offsets[, 2L]

  nn <- RANN::nn2(treelocs[, 1:2L, drop = FALSE], xyz_shifted, k = 1L)
  as.integer(nn$nn.idx[, 1L])
}

# --------------------------------------------------------------------------
# treeisonet_run_crownoff  (TLS / UAV scene — crown assignment after stems)
# --------------------------------------------------------------------------

# Port of crownOff.mergeshift().  Shifts ALL points (both already-assigned
# stem points and unassigned crown/foliage points) by their predicted 3D
# offset, then assigns each unassigned point to the nearest already-assigned
# stem point in 3D.  Assigned points keep their existing tree ID unchanged.
#
# @param shifted_xyz  numeric matrix (n,3): original XYZ + predicted offsets.
# @param stem_cls     integer vector (n): 1 = already assigned, 0 = unassigned.
# @param tree_ids     integer vector (n): existing tree IDs (0 for unassigned).
# @return integer vector (n): tree IDs; unassigned points now inherit the ID
#   of their nearest shifted stem point.
.crownoff_mergeshift <- function(shifted_xyz, stem_cls, tree_ids) {
  seg_labels <- integer(nrow(shifted_xyz))
  stem_ind   <- stem_cls > 0L
  if (!any(stem_ind)) return(seg_labels)
  seg_labels[stem_ind] <- tree_ids[stem_ind]
  if (!any(!stem_ind))  return(seg_labels)
  nn <- RANN::nn2(shifted_xyz[ stem_ind, 1:3L, drop = FALSE],
                  shifted_xyz[!stem_ind, 1:3L, drop = FALSE], k = 1L)
  seg_labels[!stem_ind] <- tree_ids[stem_ind][nn$nn.idx[, 1L]]
  seg_labels
}

# Port of crownOff.mergeremain().  After mergeshift there may still be
# isolated points whose shifted position landed far from any stem (e.g.,
# occluded upper crown).  Voxel-downsample at dec_res metres, fill
# remaining unassigned voxels by nearest-neighbour from assigned voxels,
# then propagate labels back to the full cloud.
#
# @param xyz          numeric matrix (n,3): ORIGINAL (un-shifted) XYZ.
# @param init_labels  integer vector (n): output of .crownoff_mergeshift().
# @param dec_res      numeric: voxel edge for downsampling before NN fill (m).
# @return integer vector (n): fully-filled tree IDs.
.crownoff_mergeremain <- function(xyz, init_labels, dec_res = 0.2) {
  xyz_min  <- apply(xyz, 2L, min)
  vox      <- floor((xyz - rep(xyz_min, each = nrow(xyz))) / dec_res)
  vox_key  <- paste(vox[, 1L], vox[, 2L], vox[, 3L], sep = ":")
  u_idx    <- which(!duplicated(vox_key))   # one representative per voxel
  grp_idx  <- match(vox_key, vox_key[u_idx])  # inverse map back to full cloud

  xyz_dec    <- xyz[u_idx, 1:3L, drop = FALSE]
  label_dec  <- init_labels[u_idx]
  seg_labels <- integer(length(u_idx))

  exist_ind <- label_dec > 0L
  if (any(exist_ind) && any(!exist_ind)) {
    nn <- RANN::nn2(xyz_dec[ exist_ind, , drop = FALSE],
                    xyz_dec[!exist_ind, , drop = FALSE], k = 1L)
    seg_labels[!exist_ind] <- label_dec[exist_ind][nn$nn.idx[, 1L]]
  }
  seg_labels[exist_ind] <- label_dec[exist_ind]
  seg_labels[grp_idx]
}

#' Assign tree IDs to crown/foliage points using the CrownOff regression model.
#'
#' R port of \code{crownOff.crownOff()} from TreeAIBox.  This is the TLS/UAV
#' equivalent of \code{treeisonet_run_treeoff()}, used as step 4 in the
#' TreeisoNet pipeline after \code{treeisonet_shortestpath3D()} has assigned
#' IDs to stem points only.
#'
#' The key difference from the ALS treeOff model is \strong{channel 1}:
#' \itemize{
#'   \item treeOff (ALS): channel 1 = 2D treeloc indicator broadcast along Z.
#'   \item crownOff (TLS): channel 1 = per-3D-voxel mean of \code{stem_cls}
#'         (binary flag marking voxels that contain already-assigned stem
#'         points from \code{shortestpath3D}).  This gives the model a 3D
#'         map of existing stem segments as context for predicting where each
#'         crown point belongs.
#' }
#' After inference the function applies two post-processing steps that also
#' differ from treeOff:
#' \enumerate{
#'   \item \strong{mergeshift}: shift every point by its predicted 3D
#'         (dx,dy,dz) offset, then assign each unassigned point to the
#'         nearest already-assigned (stem) point in 3D.
#'   \item \strong{mergeremain}: fill any still-unassigned points by
#'         nearest-neighbour after 0.2 m voxel downsampling.
#' }
#'
#' @param xyz          numeric matrix (n, 3) of all point XYZ (full cloud,
#'   including already-assigned stem points).
#' @param tree_ids     integer vector (n): per-point tree ID from
#'   \code{treeisonet_shortestpath3D()}; 0 = unassigned (foliage/crown).
#' @param model_bundle list from \code{load_treeaibox_model()} for a
#'   \code{crownoff} model (\code{head_type="regression"}, \code{out_chans=3}).
#' @param vox_override optional list with named elements \code{xy} and/or
#'   \code{z} to override the model's voxel resolution (metres).
#' @param verbose      print per-block progress bar.
#' @return integer vector (n): tree IDs for all points.  Stem-point IDs from
#'   \code{tree_ids} are preserved unchanged; crown points receive IDs via
#'   mergeshift + mergeremain.  Residual zeros are rare (truly isolated pts).
#' @export
treeisonet_run_crownoff <- function(xyz, tree_ids, model_bundle,
                                    vox_override = NULL, verbose = FALSE) {
  if (!requireNamespace("RANN", quietly = TRUE))
    stop("treeisonet_run_crownoff() requires the RANN package.")

  model    <- model_bundle$model
  device   <- model_bundle$device
  cfg      <- model_bundle$config$model
  nbmat_sz <- as.integer(cfg$voxel_number_in_block)   # c(X, Y, Z)
  min_res  <- as.numeric(cfg$voxel_resolution_in_meter)
  if (!is.null(vox_override$xy) && !is.na(vox_override$xy))
    min_res[1:2L] <- as.numeric(vox_override$xy)
  if (!is.null(vox_override$z) && !is.na(vox_override$z))
    min_res[3L] <- as.numeric(vox_override$z)
  nb_tsz <- prod(nbmat_sz)

  # Binary stem flag: 1 = already assigned by shortestpath3D, 0 = crown/foliage.
  # This is the context signal fed as channel 1 to the model.
  stem_cls <- as.integer(tree_ids > 0L)

  pcd_min <- apply(xyz, 2L, min)

  # 2D XY block grouping (same partitioning as treeOff)
  block_ij <- floor(
    (xyz[, 1:2L, drop = FALSE] - rep(pcd_min[1:2L], each = nrow(xyz))) /
    (min_res[1:2L] * nbmat_sz[1:2L])
  )
  block_key    <- paste(block_ij[, 1L], block_ij[, 2L], sep = "_")
  block_groups <- split(seq_len(nrow(xyz)), block_key)

  pcd_pred <- matrix(0, nrow(xyz), 3L)   # per-point (dx, dy, dz) in voxel units

  nblk <- length(block_groups)
  .pb  <- if (verbose) {
    function(i) invisible(NULL)
  } else {
    function(i) {
      filled <- round(30L * i / nblk)
      cat(sprintf("\r  [%s] %3d%%  (%d/%d blocks)",
                  paste0(strrep("=", filled), strrep(" ", 30L - filled)),
                  round(100L * i / nblk), i, nblk),
          file = stderr())
      if (i == nblk) cat("\n", file = stderr())
    }
  }

  model$eval()
  for (k in seq_along(block_groups)) {
    .pb(k)
    idx       <- block_groups[[k]]
    pts_block <- xyz[idx, 1:3L, drop = FALSE]
    sp_min    <- apply(pts_block, 2L, min)

    # Voxelize block → flat 3D indices (X-major order, same as treeOff)
    ijk <- floor((pts_block - rep(sp_min, each = nrow(pts_block))) /
                 rep(min_res, each = nrow(pts_block)))
    valid <- apply(ijk < rep(nbmat_sz, each = nrow(ijk)) & ijk >= 0L, 1L, all)
    ijk_v <- ijk[valid, , drop = FALSE]
    idx_v <- idx[valid]
    if (length(idx_v) == 0L) next

    flat3d <- as.integer(ijk_v[, 1L]) * nbmat_sz[2L] * nbmat_sz[3L] +
              as.integer(ijk_v[, 2L]) * nbmat_sz[3L] +
              as.integer(ijk_v[, 3L])
    unq3d <- unique(flat3d)
    inv3d <- match(flat3d, unq3d)

    # Channel 1: per-3D-voxel mean of stem_cls for points in this block.
    # Port of: npg.aggregate(nb_inverse_idx, stem_cls[idx][nb_sel], 'mean')
    nb_stem_u <- as.numeric(tapply(stem_cls[idx_v], inv3d, mean))

    # Build 2-channel input tensor: (nb_tsz, 2) → (1, 2, Z, Y, X)
    x_data <- torch_zeros(nb_tsz, 2L)
    x_data[unq3d + 1L, 1L] <- 1.0              # channel 0: occupancy
    x_data[unq3d + 1L, 2L] <- nb_stem_u        # channel 1: stem fraction per voxel
    x_in <- x_data$
      reshape(c(1L, nbmat_sz[1L], nbmat_sz[2L], nbmat_sz[3L], 2L))$
      permute(c(1L, 5L, 2L, 3L, 4L))$           # (B, C=2, X, Y, Z)
      transpose(3L, 5L)$                         # (B, C=2, Z, Y, X)
      to(device = device)

    # Forward → h: (B=1, out_chans=3, Z, Y, X)
    with_no_grad({ h <- model(x_in) })

    # Extract 3D offsets at occupied voxels, propagate to per-point predictions.
    # Port of: torch.swapaxes(h,-1,2) → moveaxis(1,-1) → reshape(nb_tsz,3)
    #                                                    → [idx] → inverse map
    h_flat      <- h$transpose(3L, 5L)$            # (B, 3, X, Y, Z)
                    permute(c(1L, 3L, 4L, 5L, 2L))$# (B, X, Y, Z, 3)
                    reshape(c(nb_tsz, 3L))          # (nb_tsz, 3)
    pred_at_pts <- as.matrix(h_flat[unq3d + 1L, ]$cpu())  # (n_unq, 3)
    pcd_pred[idx_v, ] <- pred_at_pts[inv3d, ]      # broadcast back to all points
  }

  # Scale voxel-unit offsets to metres
  pcd_pred[, 1L] <- pcd_pred[, 1L] * min_res[1L]
  pcd_pred[, 2L] <- pcd_pred[, 2L] * min_res[2L]
  pcd_pred[, 3L] <- pcd_pred[, 3L] * min_res[3L]

  # Shift full cloud by predicted offsets (stems shift too, but keep their IDs)
  shifted_xyz <- xyz[, 1:3L, drop = FALSE] + pcd_pred

  # Step 1 — mergeshift: assign each unassigned point to nearest shifted stem
  init_labels <- .crownoff_mergeshift(shifted_xyz, stem_cls, tree_ids)

  # Step 2 — mergeremain: NN-fill any residual zeros using original coords
  as.integer(.crownoff_mergeremain(xyz[, 1:3L, drop = FALSE], init_labels,
                                   dec_res = 0.2))
}
