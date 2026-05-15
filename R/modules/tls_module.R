# ============================================================================
# TLS MODULE: Terrestrial Laser Scanner Processing
# ============================================================================
# Pipeline:
#   1. Read + inspect cloud (lidR readLAS / las_check).
#   2. Pre-process: dedupe, SOR, then decimate to scanner precision
#      (res = 0.02 m, RS10). No ground classification, height normalization,
#      or height filter needed: all DL models voxelize per-block relative to
#      local block minimum, so they are invariant to absolute Z values.
#   3. Segmentation: TreeFilter -> overstory mask; TreeisoNet (StemCls ->
#      TreeLoc -> shortestpath3D) -> TreeID per overstory point.
#   4. WoodCls: wood/foliage DL classification -> WoodLabel per point.
#      4.1 Branch/foliage classification (full cloud).
#      4.2 Downed log detection (near-ground 2D WoodCls + PCA tilt filter).
#      4.3 8-class forest component labelling (Xi 2023).
#   5. Per-tree QC PNGs: four-panel RGB + forest-component side views per TreeID.
#   6. QSM: cut-pursuit over-segmentation + Dijkstra skeleton + cylinder fit.
#
# Refs:
#   Tao et al. 2015 ISPRS J. Photogramm. Remote Sens. 110:66-76
#   Larysch et al. 2025 doi:10.1007/s10342-025-01796-z
# ============================================================================


# ---- Packages + parallelism ------------------------------------------------

library(lidR)
library(sf)
library(treeAIBoxR)
# Xeon w5-2545: 12 physical / 24 logical cores, 128 GB RAM. No competition
# for cycles on this box, so use all logical threads.
n_threads <- parallel::detectCores(logical = TRUE)   # 24
lidR::set_lidr_threads(n_threads)
if (requireNamespace("data.table", quietly = TRUE))
  data.table::setDTthreads(n_threads)
# BLAS/LAPACK and Rcpp+OpenMP thread cap is set in TILE/.Renviron via
# OMP_NUM_THREADS (also governs OPENBLAS/MKL) and applied at R startup.
# Restart R after changing it.
options(lidR.progress = TRUE)


# ============================================================================
# CONFIG
# ============================================================================

tls_input <- "C:/Users/AGRIST/OneDrive - Government of BC/Sklar, Daniel FOR_EX's files - Great Beaver Lake/Plot 311/20250611101844578.las"
out_dir   <- "G:/alpine_treemapping/TILE/outputs/tls/Plot_311"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Plot geometry — used in Section 3 trim and Section 5 QC PNGs.
# Trees outside (plot_radius + plot_buffer) are dropped after segmentation.
plot_center_x <- 0.0     # m; scanner-local frame
plot_center_y <- 0.0
plot_radius   <- 11.28   # m
plot_buffer   <- 1.72    # m; extra edge margin (-> 13 m total)

# Per-tree QC PNG output directory
tree_qc_subdir <- "tree_qc"

# Post-segmentation tree quality filters (applied in Section 3.3)
min_tree_height_m <- 6.0    # m; drop trees with Z range < this
min_tree_points   <- 200L   # pts; drop trees with fewer total points



# 1.0 READ----


las_raw <- readLAS(tls_input, select = "xyzicrnRGB")
if (is.empty(las_raw)) stop("Input LAS is empty / failed to read.")
las_check(las_raw)

# 2.0 PRE-PROCESSING----

# 2.1 Point Deduplication
las_preproc <- filter_duplicates(las_raw)

# 2.2 Statistical outlier removal
npts0 <- npoints(las_preproc)
las_preproc <- classify_noise(las_preproc, sor(k = 8, m = 3))
las_preproc <- filter_poi(las_preproc, Classification != LASNOISE)
message(sprintf("SOR: removed %d / %d points (%.2f%%) as noise.",
                npts0 - npoints(las_preproc), npts0,
                100 * (npts0 - npoints(las_preproc)) / npts0))

# 2.3 Point decimation (3D) — one random point per 2 cm voxel.
# RS10 SLAM inter-frame error ~2 cm; no sub-2cm geometry is recoverable.
npts1 <- npoints(las_preproc)
las_preproc <- decimate_points(las_preproc, random_per_voxel(res = 0.02))
message(sprintf("Decimate: %d -> %d points (%.1f%% retained).",
                npts1, npoints(las_preproc), 100 * npoints(las_preproc) / npts1))

writeLAS(las_preproc, file.path(out_dir, "checkpoint_01_preproc.laz"))

# 3. SEGMENT ----

# 3.1 TreeFiltering: classify each point as overstory (2) or understory/ground (1).
las_filtered <- treeAIBoxR::treefiltering(
  las_preproc,
  model   = "treefiltering_tls_esegformer3D_128_8cm(GPU3GB)",
  cuda    = TRUE,
  verbose = TRUE
)
# QC: plot_cloud_qc(las_filtered, color_by = "TreeFilterLabel")
writeLAS(las_filtered, file.path(out_dir, "checkpoint_02a_treefilter.laz"))


# 3.2 TreeisoNet: StemCls -> TreeLoc -> shortestpath3D -> CrownOff3D -> CrownClustersSP.
#     Adds TreeID, StemCls, TreeLocX, TreeLocY. Runs on overstory points only.
las_segmented <- treeAIBoxR::treeisonet(
  las_filtered,
  stemcls_model     = "treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)",
  treeloc_model     = "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)",
  crownoff_model    = "treeisonet_tls_boreal_crownoff_esegformer3D_128_15cm(GPU4GB)",
  conf_thresh       = 0.1,   # default 0.3 - minimum; catch even faint stem responses
  nms_thresh_xy     = 0.15,  # default 0.5 m - 0.05 caused phantom double-detection splits; 0.15 = 1x stem diameter
  max_isolated_dist = 0.12,  # default 0.3 m - 3x voxel size; bridges stem/crown gaps without jumping trees
  k_graph           = 5L,    # default 10   - vertical redundancy without reopening cross-crown wiring
  cuda              = TRUE,
  verbose           = TRUE
)

# 3.3 Trim: drop trees whose TreeLoc base falls outside plot_radius + plot_buffer.
#     After trim, las contains only points assigned to plot trees.
r_max    <- plot_radius + plot_buffer
plot_ids <- las_segmented@data[TreeID > 0L & !is.na(TreeLocX),
                     .(lx = TreeLocX[[1L]], ly = TreeLocY[[1L]]), by = TreeID
                    ][sqrt((lx - plot_center_x)^2 + (ly - plot_center_y)^2) <= r_max,
                      TreeID]
las_segmented <- filter_poi(las_segmented, TreeID %in% plot_ids)
message(sprintf("[3] Plot filter: %d trees within %.2f m of center.", length(plot_ids), r_max))

# 3.4 Height filter: drop trees whose Z span (max-min) is below min_tree_height_m.
tree_heights <- las_segmented@data[TreeID > 0L,
                         .(height = max(Z) - min(Z)), by = TreeID]
hgt_ids <- tree_heights[height >= min_tree_height_m, TreeID]
las_segmented <- filter_poi(las_segmented, TreeID %in% hgt_ids)
message(sprintf("[3] Height filter: %d trees >= %.1f m (dropped %d).",
                length(hgt_ids), min_tree_height_m,
                length(plot_ids) - length(hgt_ids)))

# 3.5 Point-count filter: drop trees with fewer than min_tree_points total points.
tree_counts <- las_segmented@data[TreeID > 0L, .N, by = TreeID]
pt_ids <- tree_counts[N >= min_tree_points, TreeID]
las_segmented <- filter_poi(las_segmented, TreeID %in% pt_ids)
message(sprintf("[3] Point filter: %d trees >= %d pts (dropped %d).",
                length(pt_ids), min_tree_points,
                length(hgt_ids) - length(pt_ids)))

writeLAS(las_segmented, file.path(out_dir, "checkpoint_02b_segmented.laz"))

offsets <- plot(las_segmented, color = "TreeID", pal = sample(rainbow(100)))
.tops   <- las_segmented@data[TreeID > 0, .(x = TreeLocX[1], y = TreeLocY[1], z = max(Z)), keyby = TreeID]
rgl::text3d(.tops$x - offsets[1], .tops$y - offsets[2], .tops$z + 0.3, texts = as.character(.tops$TreeID), col = "white", cex = 0.8)

# QC: per-tree segmentation PNGs (side views coloured by TreeID in neighbourhood)
# Each PNG shows all points within hw_seg of the tree centre — colour = TreeID.
# Splits appear as stacked colour bands; neighbour bleed-in appears on the sides.
if (TRUE) {   # set FALSE to skip
  seg_qc_dir <- file.path(out_dir, "seg_qc")
  if (dir.exists(seg_qc_dir))
    invisible(file.remove(list.files(seg_qc_dir, pattern = "\\.png$", full.names = TRUE)))
  dir.create(seg_qc_dir, showWarnings = FALSE, recursive = TRUE)

  data.table::setDT(las_segmented@data)
  seg_pts  <- las_segmented@data[!is.na(TreeID) & TreeID > 0L]
  seg_ids  <- sort(unique(seg_pts$TreeID))

  # Focal tree gets a unique colour; all other neighbourhood points = dark grey.
  set.seed(42)
  focal_col <- setNames(sample(rainbow(length(seg_ids), s = 0.9, v = 0.9)),
                        as.character(seg_ids))

  hw_seg   <- 3.0   # m half-width of neighbourhood shown in each panel

  message(sprintf("[seg_qc] Saving %d per-tree PNGs to %s", length(seg_ids), seg_qc_dir))
  for (id in seg_ids) {
    focal  <- seg_pts[TreeID == id]
    cx     <- focal$TreeLocX[1]; cy <- focal$TreeLocY[1]   # centre on stem base

    # All points (any TreeID) in the neighbourhood for context
    hood   <- seg_pts[abs(X - cx) <= hw_seg & abs(Y - cy) <= hw_seg]
    cols   <- ifelse(hood$TreeID == id, focal_col[as.character(id)], "#404040")

    z_rng  <- range(focal$Z, na.rm = TRUE)
    z_rng  <- c(z_rng[1] - 0.2, z_rng[2] + 0.2)

    png(file.path(seg_qc_dir, sprintf("seg_%04d.png", id)),
        width = 2100, height = 750, bg = "black", res = 110)
    op <- par(mfrow = c(1, 3), bg = "black", fg = "white",
              col.axis = "white", col.lab = "white", col.main = "white",
              mar = c(4, 4, 3, 1))

    plot(hood$X - cx, hood$Z, pch = ".", cex = 0.7, col = cols,
         xlab = "Stem Base X offset (m)", ylab = "Height (m)",
         main = sprintf("TreeID %d  XZ  pts=%d  stem=(%.1f, %.1f)", id, nrow(focal), cx, cy),
         xlim = c(-hw_seg, hw_seg), ylim = z_rng)
    abline(v = 0, col = "white", lty = 2, lwd = 0.8)

    plot(hood$Y - cy, hood$Z, pch = ".", cex = 0.7, col = cols,
         xlab = "Stem Base Y offset (m)", ylab = "Height (m)",
         main = sprintf("TreeID %d  YZ", id),
         xlim = c(-hw_seg, hw_seg), ylim = z_rng)
    abline(v = 0, col = "white", lty = 2, lwd = 0.8)

    plot(hood$X - cx, hood$Y - cy, pch = ".", cex = 0.7, col = cols,
         xlab = "Stem Base X offset (m)", ylab = "Stem Base Y offset (m)",
         main = sprintf("TreeID %d  XY (top-down)", id),
         xlim = c(-hw_seg, hw_seg), ylim = c(-hw_seg, hw_seg), asp = 1)
    abline(v = 0, h = 0, col = "white", lty = 2, lwd = 0.8)

    par(op); dev.off()
  }
  message("[seg_qc] Done: ", seg_qc_dir)
}


# ============================================================================
# 4. WoodCls  (wood / foliage classification)
# ============================================================================
las_woodcls <- treeAIBoxR::woodcls(
  las_segmented,
  model     = "woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)",
  sensor    = "tls",
  component = "branch",
  cuda      = TRUE,
  verbose   = TRUE
)

las_woodcls <- treeAIBoxR::woodcls(
  las_woodcls,
  model               = "woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)",
  component           = "downed_log",
  downed_z_max        = 1.5,    # m above ground (Xi 2023)
  downed_tilt_min_deg = 60.0,   # deg from vertical (Xi 2023)
  cuda                = TRUE,
  verbose             = TRUE
)

las_woodcls <- treeAIBoxR::forest_components(las_woodcls)

writeLAS(las_woodcls, file.path(out_dir, "checkpoint_03_woodcls.laz"))
# QC: xi_pal <- c("0"="#aaff01","1"="#f02b00","2"="#7a4101","3"="#0a9e00","4"="#9f0aef","5"="#f54b8c","6"="#ae5504","7"="#0000fe")
# QC: plot(las, color = "ComponentClass", colorPalette = xi_pal)


# ============================================================================
# 5. Per-tree QC PNGs
# ============================================================================
# Four-panel side views per TreeID written to <out_dir>/<tree_qc_subdir>/.
# Top row: RGB XZ / RGB YZ (or component colours if no RGB).
# Bottom row: forest_components() colouring (8-class Xi palette).
# Not all classes will appear per tree -- legend is dynamic.
# Runs in batch (builds qc_las locally; does not require Section 7 rgl).
if (TRUE) {   # set to FALSE to skip QC PNGs
  qc_dir <- file.path(out_dir, tree_qc_subdir)
  if (dir.exists(qc_dir))
    invisible(file.remove(list.files(qc_dir, pattern = "\\.png$", full.names = TRUE)))
  dir.create(qc_dir, showWarnings = FALSE, recursive = TRUE)
  qc_las <- lidR::filter_poi(las_woodcls, !is.na(TreeID) & TreeID > 0L)
  # ComponentClass already present from Section 4 forest_components() pipeline step.
  data.table::setDT(qc_las@data)
  qc_ids <- sort(unique(qc_las@data$TreeID))
  qc_ids <- qc_ids[!is.na(qc_ids)]
  hw <- 3.0   # m; half-width of side-view panels
  has_rgb <- all(c("R", "G", "B") %in% names(qc_las@data)) &&
             { rgb_max <- max(qc_las@data$R, qc_las@data$G, qc_las@data$B,
                              na.rm = TRUE)
               isTRUE(is.finite(rgb_max) && rgb_max > 0) }
  rgb_div <- if (has_rgb) {
    if (max(qc_las@data$R, qc_las@data$G, qc_las@data$B, na.rm = TRUE) > 256) 65535
    else 255
  } else NA_real_
  fc_pal   <- c("0" = "#aaff01", "1" = "#f02b00", "2" = "#7a4101",
               "3" = "#0a9e00", "4" = "#9f0aef", "5" = "#f54b8c",
               "6" = "#ae5504", "7" = "#0000fe")
  fc_names <- c("0" = "Grass/remaining", "1" = "Stem",    "2" = "Branch",
               "3" = "Foliage",        "4" = "Downed log", "5" = "Sapling stem",
               "6" = "Below-canopy br.", "7" = "Ground")
  message(sprintf("Saving %d per-tree QC PNGs to %s%s",
                  length(qc_ids), qc_dir,
                  if (has_rgb) " [RGB + component]" else " [component]"))
  for (id in qc_ids) {
    pts <- qc_las@data[TreeID == id]
    if (nrow(pts) < 50L) next
    cx     <- median(pts$X); cy <- median(pts$Y)
    ylim_z <- range(pts$Z, na.rm = TRUE)   # full tree height, not clipped by asp
    if (has_rgb) {
      r8 <- pmin(pmax(pts$R / rgb_div, 0), 1)
      g8 <- pmin(pmax(pts$G / rgb_div, 0), 1)
      b8 <- pmin(pmax(pts$B / rgb_div, 0), 1)
      rgb_col <- grDevices::rgb(r8, g8, b8)
      fc_lbl <- pts$ComponentClass
      fc_col <- fc_pal[as.character(fc_lbl)]
      fc_col[is.na(fc_col)] <- "#aaff01"
      present <- as.character(sort(unique(fc_lbl)))
      png(file.path(qc_dir, sprintf("tree_%04d.png", id)),
          width = 1600, height = 1400, bg = "black", res = 110)
      op <- par(mfrow = c(2, 2), bg = "black", fg = "white",
                col.axis = "white", col.lab = "white", col.main = "white",
                mar = c(4, 4, 3, 1))
      plot(pts$X - cx, pts$Z, pch = ".", cex = 0.7, col = rgb_col,
           xlab = "X offset (m)", ylab = "Z (m)",
           main = sprintf("TreeID %d  RGB XZ  (look along Y)", id),
           xlim = c(-hw, hw), ylim = ylim_z)
      plot(pts$Y - cy, pts$Z, pch = ".", cex = 0.7, col = rgb_col,
           xlab = "Y offset (m)", ylab = "Z (m)",
           main = sprintf("RGB YZ  pts=%d", nrow(pts)),
           xlim = c(-hw, hw), ylim = ylim_z)
      plot(pts$X - cx, pts$Z, pch = ".", cex = 0.7, col = fc_col,
           xlab = "X offset (m)", ylab = "Z (m)",
           main = "Forest components XZ",
           xlim = c(-hw, hw), ylim = ylim_z)
      legend("topright", legend = fc_names[present], col = fc_pal[present],
             pch = 16, cex = 0.55, bg = "black", text.col = "white",
             bty = "o", box.col = "grey40")
      plot(pts$Y - cy, pts$Z, pch = ".", cex = 0.7, col = fc_col,
           xlab = "Y offset (m)", ylab = "Z (m)",
           main = "Forest components YZ",
           xlim = c(-hw, hw), ylim = ylim_z)
      par(op); dev.off()
    } else {
      fc_lbl <- pts$ComponentClass
      fc_col <- fc_pal[as.character(fc_lbl)]
      fc_col[is.na(fc_col)] <- "#aaff01"
      present <- as.character(sort(unique(fc_lbl)))
      png(file.path(qc_dir, sprintf("tree_%04d.png", id)),
          width = 1400, height = 900, bg = "black", res = 110)
      op <- par(mfrow = c(1, 2), bg = "black", fg = "white",
                col.axis = "white", col.lab = "white", col.main = "white",
                mar = c(4, 4, 3, 1))
      plot(pts$X - cx, pts$Z, pch = ".", cex = 0.6, col = fc_col,
           xlab = "X offset (m)", ylab = "Z (m)",
           main = sprintf("TreeID %d  XZ  pts=%d", id, nrow(pts)),
           xlim = c(-hw, hw), ylim = ylim_z)
      legend("topright", legend = fc_names[present], col = fc_pal[present],
             pch = 16, cex = 0.55, bg = "black", text.col = "white",
             bty = "o", box.col = "grey40")
      plot(pts$Y - cy, pts$Z, pch = ".", cex = 0.6, col = fc_col,
           xlab = "Y offset (m)", ylab = "Z (m)",
           main = "YZ view",
           xlim = c(-hw, hw), ylim = ylim_z)
      par(op); dev.off()
    }
  }
  message("Per-tree QC done: ", qc_dir)
}


# ============================================================================
# 6. QSM (Quantitative Structure Model)
# ============================================================================
# Runs cut-pursuit over-segmentation, Dijkstra skeleton tracing, and
# Kasa circle-fit radius estimation per tree.  Requires TreeID, WoodLabel,
# and StemCls (produced by treeisonet + woodcls above).
# Output: one row per cylinder with branchID, parentBranchID, branchOrder,
# start/end XYZ, radius (m), length (m).

cylinders <- treeAIBoxR::qsm(
  las_woodcls,
  verbose = TRUE
)

if (!is.null(cylinders)) {
  cyl_path <- file.path(out_dir, "qsm_cylinders.csv")
  write.csv(cylinders, cyl_path, row.names = FALSE)
  message(sprintf("[QSM] %d cylinders across %d trees written to %s",
                  nrow(cylinders),
                  length(unique(cylinders$TreeID)),
                  cyl_path))
} else {
  warning("[QSM] No cylinders produced — check TreeID, WoodLabel, StemCls fields.")
}
