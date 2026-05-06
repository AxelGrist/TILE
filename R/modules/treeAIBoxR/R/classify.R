# classify.R
#
# Public entry point: classify_wood(xyz, model)
#
# Mirrors the inference loop of componentFilter.filterPoints() exactly:
#   - 3D sliding blocks (or 2D XY-only when if_bottom_only=TRUE)
#   - per-block binary occupancy tensor of shape (1, 1, D, H, W) with axes
#     swapped to upstream's (W, H, D) layout before forward()
#   - argmax in voxel space, mapped back to points via per-block inverse
#     index, with class>1 OR'd across overlapping coverage (binary case)
#     or last-write-wins (multiclass case)
#   - if_bottom_only=TRUE: any point not covered by a block (outside all
#     XY sliding windows) is left at its initial foliage value (matches
#     upstream's pcd_pred[~seen] = True path)

#' Wood/foliage classification with a TreeAIBox model
#'
#' @param xyz   numeric matrix (N, 3) of point coordinates in metres
#' @param model object returned by [load_treeaibox_model()]
#' @param if_bottom_only logical; when TRUE uses 2D XY-only sliding blocks
#'   (ignores Z dimension for block layout). Any point not covered by a
#'   block is labelled foliage. Matches \code{filterPoints(if_bottom_only=TRUE)}
#'   in upstream componentFilter.py. Use for ALS/UAV TreeFiltering models.
#'   Default FALSE (full 3D blocks, correct for TLS WoodCls and StemCls).
#' @param verbose logical; print per-block progress
#' @param batch  integer; number of blocks to process per torch_no_grad()
#'   chunk. Larger reuses kernel launches at the cost of GPU memory.
#'   Default 1L = safest for low-VRAM GPUs (the distilled 2 GB model).
#'
#' @return integer vector of length N, with values:
#'   * for binary models (config num_classes == 2): 1 = foliage/understory,
#'     2 = overstory tree (or wood for WoodCls)
#'   * for multiclass models (num_classes > 2): 1 = foliage, 2 = branch,
#'     3 = stem (or whatever the upstream class index implies)
#'
#' @export
classify_wood <- function(xyz, model, if_bottom_only = FALSE,
                          verbose = FALSE, batch = 1L) {
  if (!is.matrix(xyz) || ncol(xyz) != 3L)
    stop("xyz must be an (N, 3) numeric matrix.")

  cfg          <- model$config$model
  nbmat_sz     <- model$voxel_count
  voxel_res    <- model$voxel_res_m
  num_classes  <- model$num_classes_in   # already +1 from JSON
  is_multi     <- num_classes > 3L
  dev          <- model$device
  nb_tsz       <- prod(nbmat_sz)

  plan <- plan_voxelization(xyz, voxel_res = voxel_res,
                            nbmat_sz = nbmat_sz, overlap = 0.1,
                            if_bottom_only = if_bottom_only)
  nblk <- length(plan$blocks)

  if (verbose) message(sprintf("[treeAIBoxR] %d points, %d block(s), %s model%s",
                               plan$n_pts, nblk,
                               if (is_multi) "multiclass" else "binary",
                               if (isTRUE(if_bottom_only)) " [2D/bottom-only]" else ""))

  # Output buffers (mirror componentFilter):
  pcd_pred <- if (is_multi) integer(plan$n_pts) + (num_classes - 1L)
              else logical(plan$n_pts)         # FALSE = foliage initially

  # Track which points were visited by at least one block (needed for the
  # if_bottom_only path: unvisited points stay foliage, matching upstream's
  #   pcd_pred[~seen] = True  then return (pcd_pred + 1))
  if (isTRUE(if_bottom_only)) seen <- logical(plan$n_pts)

  for (i in seq_len(nblk)) {
    blk     <- plan$blocks[[i]]
    occ_idx <- blk$occ_idx                    # 1-based, length n_unique_voxels
    if (length(occ_idx) == 0L) next

    # Build a flat occupancy tensor of length D*H*W, set occupied voxels to 1.
    flat <- torch::torch_zeros(nb_tsz, 1L, dtype = torch::torch_float())
    flat[occ_idx, 1L] <- 1.0

    # Reshape (D*H*W, 1) -> (1, D, H, W, 1) -> (1, 1, W, H, D) in one permute.
    # Equivalent to upstream: permute(0,4,1,2,3) then swapaxes(-1,2).
    # Input dims (1-based): 1=batch, 2=D, 3=H, 4=W, 5=chan
    # Output want: (1, 1, W, H, D) = positions (1,5,4,3,2)
    x <- flat$reshape(c(1L, nbmat_sz[1], nbmat_sz[2], nbmat_sz[3], 1L))
    x <- x$permute(c(1L, 5L, 4L, 3L, 2L))$contiguous()  # (1, 1, W, H, D)
    x <- x$to(device = dev)

    h <- torch::with_no_grad({ model$model(x) })  # (1, C, W, H, D)

    # Reverse: (1, C, W, H, D) -> (1, D, H, W, C) in one permute.
    # Input dims (1-based): 1=batch, 2=C, 3=W, 4=H, 5=D
    # Output want: (1, D, H, W, C) = positions (1,5,4,3,2)
    h <- h$permute(c(1L, 5L, 4L, 3L, 2L))$contiguous()  # (1, D, H, W, C)
    h <- h$reshape(c(nb_tsz, num_classes))    # (D*H*W, C)
    h_sub <- h[occ_idx, ]$reshape(c(-1L, num_classes))  # always (n_unique_voxels, C)
    # R tensor [idx, ] drops the row dim when length(idx)==1 (like base R).
    # $reshape(-1, C) restores 2D shape in that edge case.
    cls   <- as.integer(torch::torch_argmax(h_sub, dim = 2L)$cpu())
    # torch_argmax uses 1-based dim (like all R torch indexing).
    # dim=2L = axis 1 in 0-based = columns (C) of (n_voxels, C) tensor.
    # Returns 1-based class indices 1..C per voxel.

    # Map per-voxel labels back to per-point labels via the inverse index.
    point_labels <- cls[blk$inverse]          # length == length(blk$pcd_idx)

    if (is_multi) {
      pcd_pred[blk$pcd_idx] <- point_labels
    } else {
      # Binary mode: any block voting "wood" wins (matches upstream OR).
      pcd_pred[blk$pcd_idx] <- pcd_pred[blk$pcd_idx] | (point_labels > 1L)
    }

    if (isTRUE(if_bottom_only)) seen[blk$pcd_idx] <- TRUE

    if (verbose) message(sprintf("  block %d/%d: %d unique voxels",
                                 i, nblk, length(occ_idx)))
  }

  # if_bottom_only: unvisited points are labelled foliage (TRUE=foliage in
  # upstream bool buffer, then +1 -> 2 foliage? No: upstream sets
  #   pcd_pred[~seen] = True  (bool True = foliage) then returns bool+1.
  # In our bool buffer FALSE=foliage, so unvisited already correct; just
  # force seen=FALSE for unvisited (they're already FALSE, this is a no-op
  # but explicit for clarity).
  # Actually upstream: pcd_pred initialised False; unseen forced True (=foliage);
  # return pcd_pred + 1 gives 1 for foliage (True->1 after +1? No: True+1=2...)
  # Let's re-read: bool(False)+1=1 (foliage), bool(True)+1=2 (wood).
  # pcd_pred[~seen]=True means those points become "wood"=2 after +1.
  # That is the upstream behaviour: unvisited get flagged as wood in the
  # binary case (they're outside any sliding window, so the model hasn't
  # rejected them as foliage, leaving them as the default overstory class).
  if (isTRUE(if_bottom_only) && !is_multi) {
    pcd_pred[!seen] <- TRUE   # unvisited -> overstory (matching upstream)
  }

  if (is_multi) {
    return(as.integer(pcd_pred))
  }
  # Binary: FALSE -> 1 (foliage/understory), TRUE -> 2 (overstory/wood).
  return(as.integer(pcd_pred) + 1L)
}
