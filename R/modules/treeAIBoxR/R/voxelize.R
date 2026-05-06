# voxelize.R
#
# Convert XYZ point clouds into binary-occupancy voxel blocks suitable
# for input to a 3D voxel SegFormer trained by NRCan TreeAIBox.
#
# Mirrors the logic of componentFilter.py:
#   1. Slice points into overlapping 3D blocks of size voxel_count * voxel_res
#   2. Per block: ijk = floor((xyz - block_origin) / voxel_res), keep in-range
#   3. Build a binary tensor of shape (D, H, W) with occupied voxels = 1
#   4. Track per-block point indices and per-point voxel indices so labels
#      can be scattered back from voxel-space to point-space.
#
# Returns a list with everything needed by classify.R.

#' Sliding 3D blocks over a point cloud
#'
#' @param pts          numeric matrix (N, 3) of XYZ coordinates
#' @param block_size   numeric length-3, block size in metres
#' @param overlap      fractional overlap between adjacent blocks (0..1)
#' @return list with `origins` (M, 3) and `groups`, a list of integer
#'   vectors of point indices per block. Each point contributes to up to
#'   2^3 = 8 overlapping blocks (matches the upstream behaviour).
#' @keywords internal
sliding_blocks <- function(pts, block_size, overlap = 0.1) {
  if (!is.matrix(pts) || ncol(pts) != 3L)
    stop("pts must be a numeric matrix with 3 columns (X, Y, Z).")
  bs     <- as.numeric(block_size)
  stride <- bs * (1 - overlap)
  p_min  <- apply(pts, 2, min)
  p_max  <- apply(pts, 2, max)
  steps  <- floor((p_max - p_min - bs) / stride) + 1L
  steps[steps < 0L] <- 0L
  dims   <- pmax(steps + 1L, 1L)   # number of blocks per axis

  rel  <- sweep(pts, 2, p_min, "-") / matrix(stride, nrow(pts), 3, byrow = TRUE)
  idx0 <- floor(rel)
  storage.mode(idx0) <- "integer"

  # For each axis, a point may belong to its primary block or its left
  # neighbour (clipped). 2^3 = 8 candidate blocks per point.
  N <- nrow(pts)
  combos <- expand.grid(dx = 0:1, dy = 0:1, dz = 0:1)
  ix_mat <- matrix(NA_integer_, N, 8L)
  iy_mat <- matrix(NA_integer_, N, 8L)
  iz_mat <- matrix(NA_integer_, N, 8L)
  for (j in seq_len(8L)) {
    ix_mat[, j] <- pmax(0L, pmin(dims[1] - 1L, idx0[, 1] - combos$dx[j]))
    iy_mat[, j] <- pmax(0L, pmin(dims[2] - 1L, idx0[, 2] - combos$dy[j]))
    iz_mat[, j] <- pmax(0L, pmin(dims[3] - 1L, idx0[, 3] - combos$dz[j]))
  }

  # Membership test: does the point fall inside the candidate block?
  block_origin_x <- p_min[1] + ix_mat * stride[1]
  block_origin_y <- p_min[2] + iy_mat * stride[2]
  block_origin_z <- p_min[3] + iz_mat * stride[3]
  pts_x <- pts[, 1]; pts_y <- pts[, 2]; pts_z <- pts[, 3]
  cond <- (pts_x >= block_origin_x & pts_x < block_origin_x + bs[1]) &
          (pts_y >= block_origin_y & pts_y < block_origin_y + bs[2]) &
          (pts_z >= block_origin_z & pts_z < block_origin_z + bs[3])

  # Build a long table of (block_id, point_id) for every (point, candidate)
  # whose condition is TRUE.
  block_ids <- ix_mat * (dims[2] * dims[3]) + iy_mat * dims[3] + iz_mat
  block_ids_flat <- as.integer(block_ids[cond])
  point_ids_flat <- as.integer(rep.int(seq_len(N), 8L)[cond])

  ord <- order(block_ids_flat)
  block_ids_flat <- block_ids_flat[ord]
  point_ids_flat <- point_ids_flat[ord]
  unique_bids <- unique(block_ids_flat)
  groups <- split(point_ids_flat, factor(block_ids_flat, levels = unique_bids))
  names(groups) <- NULL

  ix_u <- unique_bids %/% (dims[2] * dims[3])
  rem  <- unique_bids %%  (dims[2] * dims[3])
  iy_u <- rem %/% dims[3]
  iz_u <- rem %%  dims[3]
  origins <- cbind(p_min[1] + ix_u * stride[1],
                   p_min[2] + iy_u * stride[2],
                   p_min[3] + iz_u * stride[3])

  list(origins = origins, groups = groups, dims = dims)
}

#' Voxelize a single block of points into a flat occupancy vector
#'
#' Mirrors the per-block loop in componentFilter.filterPoints():
#'   ijk = floor((columns_sp[:, :3] - sp_min) / min_res)
#'   keep ijk inside [0, nbmat_sz)
#'   nb_idx = ravel_multi_index(ijk.T, nbmat_sz)
#'   unique nb_idx + inverse mapping for label scatter-back
#'
#' @param block_pts  numeric matrix (n, 3), points falling in this block
#' @param voxel_res  numeric length-3, voxel edge length in metres
#' @param nbmat_sz   integer length-3, number of voxels per axis (D, H, W)
#' @return list:
#'   \item{occ_idx}{1-based integer indices of unique occupied voxels in
#'     the flat (D*H*W) tensor (column-major to match torch's natural layout)}
#'   \item{inverse}{1-based integer mapping of length n: which entry of
#'     occ_idx each input point belongs to (after in-range filter)}
#'   \item{kept}{1-based integer indices into block_pts of points retained
#'     after the in-range filter}
#'
#' @export
voxelize_block <- function(block_pts, voxel_res, nbmat_sz) {
  if (!is.matrix(block_pts) || ncol(block_pts) != 3L)
    stop("block_pts must be a numeric matrix with 3 columns.")
  voxel_res <- as.numeric(voxel_res)
  nbmat_sz  <- as.integer(nbmat_sz)
  if (length(voxel_res) != 3L) voxel_res <- rep(voxel_res, length.out = 3L)
  if (length(nbmat_sz)  != 3L) nbmat_sz  <- rep(nbmat_sz,  length.out = 3L)

  sp_min <- apply(block_pts, 2, min)
  ijk    <- floor(sweep(block_pts, 2, sp_min, "-") /
                  matrix(voxel_res, nrow(block_pts), 3, byrow = TRUE))
  storage.mode(ijk) <- "integer"
  in_range <- ijk[, 1] >= 0L & ijk[, 1] < nbmat_sz[1] &
              ijk[, 2] >= 0L & ijk[, 2] < nbmat_sz[2] &
              ijk[, 3] >= 0L & ijk[, 3] < nbmat_sz[3]
  ijk  <- ijk[in_range, , drop = FALSE]
  kept <- which(in_range)

  # Match upstream's np.ravel_multi_index(ijk.T, nbmat_sz) which is
  # row-major: flat = i*H*W + j*W + k. We carry that exact layout into the
  # tensor via reshape((D,H,W)) below in classify.R.
  flat <- ijk[, 1] * (nbmat_sz[2] * nbmat_sz[3]) +
          ijk[, 2] *  nbmat_sz[3] +
          ijk[, 3] + 1L                             # 1-based for R

  u    <- unique(flat)
  inv  <- match(flat, u)

  list(occ_idx = u, inverse = inv, kept = kept)
}

#' Assemble a full voxelization plan for an entire tree
#'
#' Wraps sliding_blocks() + voxelize_block() over every block. Returns a
#' compact representation classify.R can iterate over without re-doing
#' any geometric work.
#'
#' @param xyz           numeric matrix (N, 3) — full point cloud
#' @param voxel_res     numeric length-3 in metres
#' @param nbmat_sz      integer length-3 voxel counts
#' @param overlap       sliding-window overlap fraction
#' @param if_bottom_only logical; when TRUE use only XY columns for sliding
#'   blocks (2D projection mode). Matches \code{filterPoints(if_bottom_only=TRUE)}
#'   in upstream componentFilter.py. Any point not covered by a block is
#'   treated as foliage by classify_wood(). Default FALSE (full 3D blocks).
#' @return list with element `blocks` (per block: pcd_idx, occ_idx,
#'   inverse) and element `n_pts`
#' @keywords internal
plan_voxelization <- function(xyz, voxel_res, nbmat_sz, overlap = 0.1,
                              if_bottom_only = FALSE) {
  block_size <- as.numeric(voxel_res) * as.numeric(nbmat_sz)
  if (isTRUE(if_bottom_only)) {
    # 2D mode: slide only in XY; each XY column block spans the entire Z extent.
    # Mirrors componentFilter.py cut_dim=2 path:
    #   sliding_blocks_point_indices(pcd[:, :2], min_res[:2]*nbmat_sz[:2], 0.1)
    # Using Inf as Z block size causes NaN in stride arithmetic; instead use
    # z_range + 1 which guarantees exactly one Z level and includes all points.
    # voxelize_block's in_range filter then keeps only the bottom nbmat_sz[3]
    # voxels; points above that are unvisited and become overstory via
    # pcd_pred[!seen] <- TRUE in classify_wood().
    z_range <- diff(range(xyz[, 3])) + 1.0
    sb <- sliding_blocks(xyz,
                         block_size = c(block_size[1:2], z_range),
                         overlap = overlap)
  } else {
    sb <- sliding_blocks(xyz, block_size = block_size, overlap = overlap)
  }
  blocks <- vector("list", length(sb$groups))
  for (b in seq_along(sb$groups)) {
    idx_in_pcd <- sb$groups[[b]]
    vb <- voxelize_block(xyz[idx_in_pcd, , drop = FALSE],
                         voxel_res = voxel_res, nbmat_sz = nbmat_sz)
    # Subset original-point indices by `kept` so downstream lookups align.
    blocks[[b]] <- list(
      pcd_idx = idx_in_pcd[vb$kept],
      occ_idx = vb$occ_idx,
      inverse = vb$inverse
    )
  }
  list(blocks = blocks, n_pts = nrow(xyz))
}
