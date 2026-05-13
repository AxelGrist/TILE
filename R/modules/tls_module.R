# ============================================================================
# TLS MODULE: Terrestrial Laser Scanner Processing
# ============================================================================
# Pipeline:
#   1. Read + inspect cloud (lidR readLAS / las_check).
#   2. Pre-process: dedupe, SOR, CSF ground classification, normalize,
#      height filter, then decimate to scanner precision (res = 0.02 m, RS10).
#   3. Segmentation: TreeFilter -> overstory mask; TreeisoNet (StemCls ->
#      TreeLoc -> shortestpath3D) -> TreeID per overstory point.
#      3.1 Eigen geometry (fine + optional coarse / multi-scale).
#      3.2 Sub-canopy foliage filter (global or stratified, optional gates).
#      3.3 Stem base detection (shape-aware DBSCAN, plot-extent + continuity).
#      3.4 Segmentation dispatcher (treeisonet / csp / shs / nearest_base).
#   4. WoodCls: wood/foliage DL classification -> WoodLabel per point.
#   5. Per-tree QC PNGs: four-panel RGB + forest-component side views per TreeID.
#   6. Forest inventory: DBH, height, position, crown projection per tree.
#   7. Interactive plotting: 3D whole-plot + single-tree views (requires inv).
#   8. Field-data validation: buffered candidate search + multi-criteria match.
#   9. QSM: cut-pursuit over-segmentation + Dijkstra skeleton + cylinder fit.
#
# Refs:
#   https://r-lidar.github.io/lidRbook/gnd.html
#   Tao et al. 2015 ISPRS J. Photogramm. Remote Sens. 110:66-76
#   Larysch et al. 2025 doi:10.1007/s10342-025-01796-z
# ============================================================================


# ---- Packages + parallelism ------------------------------------------------

library(lidR)
library(sf)
library(Rcpp)
library(treeAIBoxR)
# Xeon w5-2545: 12 physical / 24 logical cores, 128 GB RAM. No competition
# for cycles on this box, so use all logical threads.
n_threads <- parallel::detectCores(logical = TRUE)   # 24
lidR::set_lidr_threads(n_threads)
if (requireNamespace("data.table", quietly = TRUE))
  data.table::setDTthreads(n_threads)
# BLAS/LAPACK thread cap is set in TILE/.Renviron (OMP/OPENBLAS/MKL_NUM_THREADS)
# and applied at R startup -- restart R if you change it.
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



# 1.0 READ----


las <- readLAS(tls_input, select = "xyzicrnRGB")
if (is.empty(las)) stop("Input LAS is empty / failed to read.")
las_check(las)

# 2.0 PRE-PROCESSING----

# 2.1 Point Deduplication
las <- filter_duplicates(las)

# 2.2 Statistical outlier removal
# Run before ground classification so PTD receives a clean cloud.
npts0 <- npoints(las)
las <- classify_noise(las, sor(k = 8, m = 3))
las <- filter_poi(las, Classification != LASNOISE)
message(sprintf("SOR: removed %d / %d points (%.2f%%) as noise.",
                npts0 - npoints(las), npts0,
                100 * (npts0 - npoints(las)) / npts0))

# 2.3 Ground classification (CSF)
# PTD (lidR's current recommendation) misclassifies low peripheral stem bases
# as ground seeds in small TLS plots. CSF cloth simulation is more robust here.
las <- classify_ground(las, csf(
  sloop_smooth     = TRUE,
  class_threshold  = 0.05,
  cloth_resolution = 0.4,
  rigidness        = 3L,
  time_step        = 0.65
))

# 2.4 Height Normalization
las <- normalize_height(las, tin())

# 2.5 Height Filter
las <- filter_poi(las, Z <= 50)   # m; hard ceiling after normalization

# 2.6 Point decimation (3D) — one random point per 2 cm voxel.
# RS10 SLAM inter-frame error ~2 cm; no sub-2cm geometry is recoverable.
npts1 <- npoints(las)
las <- decimate_points(las, random_per_voxel(res = 0.02))
message(sprintf("Decimate: %d -> %d points (%.1f%% retained).",
                npts1, npoints(las), 100 * npoints(las) / npts1))

writeLAS(las, file.path(out_dir, "checkpoint_01_preproc.laz"))

# 3. SEGMENT ----

# 3.1 TreeFiltering: classify each point as overstory (2) or understory/ground (1).
las <- treeAIBoxR::treefiltering(
  las,
  model   = "treefiltering_tls_esegformer3D_128_8cm(GPU3GB)",
  cuda    = TRUE,
  verbose = TRUE
)
# QC: plot_cloud_qc(las, color_by = "TreeFilterLabel")
writeLAS(las, file.path(out_dir, "checkpoint_02a_treefilter.laz"))


# 3.2 TreeisoNet: StemCls -> TreeLoc -> shortestpath3D -> CrownOff3D -> CrownClustersSP.
#     Adds TreeID, StemCls, TreeLocX, TreeLocY. Runs on overstory points only

las <- treeAIBoxR::treeisonet(
  las,
  stemcls_model  = "treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)",
  treeloc_model  = "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)",
  crownoff_model = "treeisonet_tls_boreal_crownoff_esegformer3D_128_15cm(GPU4GB)",
  cuda           = TRUE,
  verbose        = TRUE
)

# 3.3 Trim: drop trees whose TreeLoc base falls outside plot_radius + plot_buffer.
#     After trim, las contains only points assigned to plot trees.
r_max    <- plot_radius + plot_buffer
plot_ids <- las@data[TreeID > 0L & !is.na(TreeLocX),
                     .(lx = TreeLocX[[1L]], ly = TreeLocY[[1L]]), by = TreeID
                    ][sqrt((lx - plot_center_x)^2 + (ly - plot_center_y)^2) <= r_max,
                      TreeID]
las <- filter_poi(las, TreeID %in% plot_ids)
message(sprintf("[3] Plot filter: %d trees within %.2f m of center.", length(plot_ids), r_max))

writeLAS(las, file.path(out_dir, "checkpoint_02b_segmented.laz"))
# QC: plot(las, color = "TreeID")


# ============================================================================
# 4. WoodCls  (wood / foliage classification)
# ============================================================================
las <- treeAIBoxR::woodcls(
  las,
  model     = "woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)",
  sensor    = "tls",
  component = "branch",
  cuda      = TRUE,
  verbose   = TRUE
)

las <- treeAIBoxR::woodcls(
  las,
  model               = "woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)",
  component           = "downed_log",
  downed_z_max        = 1.5,    # m above ground (Xi 2023)
  downed_tilt_min_deg = 60.0,   # deg from vertical (Xi 2023)
  cuda                = TRUE,
  verbose             = TRUE
)

las <- treeAIBoxR::forest_components(las)

writeLAS(las, file.path(out_dir, "checkpoint_03_woodcls.laz"))
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
  qc_las <- lidR::filter_poi(las, !is.na(TreeID) & TreeID > 0L)
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
# ============================================================================
