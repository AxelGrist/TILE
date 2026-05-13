# ============================================================================
# TLS MODULE: Terrestrial Laser Scanner Processing
# ============================================================================
# Pipeline:
#   1. Read + inspect cloud (lidR readLAS / las_check).
#   2. Pre-process at full density: dedupe, ground (CSF), normalize, SOR.
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



# ============================================================================
# HELPERS

# ============================================================================

# plot_cloud_qc(): rgl window of the full point cloud, optionally split by
# a logical mask (red = TRUE / target, green = FALSE / rest) or colored by
# TreeID / Classification / Z.
#
#   pts        : LAS object OR data.frame/data.table with X, Y, Z.
#   mask       : optional logical vector (length = nrow(pts)). mask=TRUE
#                -> red, mask=FALSE -> green. Overrides color_by.
#   color_by   : "none" | "TreeID" | "Classification" | "Z".
#   point_size : rgl point size.
#   add        : if TRUE, draw into the active rgl scene instead of
#                opening a new one.
#   title      : optional rgl title.
plot_cloud_qc <- function(pts,
                          mask       = NULL,
                          color_by   = "none",
                          point_size = 1.0,
                          add        = FALSE,
                          title      = NULL) {
  ds <- if (inherits(pts, "LAS")) pts@data else as.data.frame(pts)
  stopifnot(all(c("X", "Y", "Z") %in% names(ds)))
  if (!is.null(mask) && length(mask) != nrow(ds))
    stop("mask length must equal number of points.")

  col <- if (!is.null(mask)) {
    ifelse(mask, "#D62728", "#2CA02C")
  } else if (color_by == "TreeID" && "TreeID" %in% names(ds)) {
    pal <- grDevices::hcl.colors(60, "Dark 3")
    ifelse(is.na(ds$TreeID), "grey80",
           pal[((ds$TreeID - 1L) %% length(pal)) + 1L])
  } else if (color_by == "WoodLabel" && "WoodLabel" %in% names(ds)) {
    ifelse(is.na(ds$WoodLabel), "grey80",
           ifelse(ds$WoodLabel >= 2L, "saddlebrown", "forestgreen"))
  } else if (color_by == "XiLabel") {
    xi_pal <- c("0" = "#aaff01",   # Grass/remaining  (yellow-green)
                "1" = "#f02b00",   # Stem             (red)
                "2" = "#7a4101",   # Branch           (brown)
                "3" = "#0a9e00",   # Foliage          (green)
                "4" = "#9f0aef",   # Downed woody log (purple)
                "5" = "#f54b8c",   # Sapling stem     (pink)
                "6" = "#ae5504",   # Below-canopy br. (orange-brown)
                "7" = "#0000fe")   # Ground           (blue)
    lbl <- if ("ComponentClass" %in% names(ds)) ds$ComponentClass
           else forest_component_labels(ds)
    xi_pal[as.character(lbl)]
  } else if (color_by == "Classification" && "Classification" %in% names(ds)) {
    ifelse(ds$Classification == 2L, "saddlebrown", "forestgreen")
  } else if (color_by == "Z") {
    grDevices::hcl.colors(64, "Viridis")[
      cut(ds$Z, breaks = 64, labels = FALSE, include.lowest = TRUE)]
  } else {
    "black"
  }

  if (!add) { rgl::open3d(); rgl::bg3d("white") }
  rgl::points3d(ds$X, ds$Y, ds$Z, color = col, size = point_size)
  rgl::aspect3d("iso")
  rgl::axes3d(c("x--", "y--", "z--"), col = "grey40")
  if (!is.null(title)) rgl::title3d(main = iconv(title, to = "ASCII//TRANSLIT"), col = "black")
  if (!is.null(mask))
    rgl::legend3d("topright",
                  legend = c("target (mask=TRUE)", "rest"),
                  col = c("#D62728", "#2CA02C"), pch = 19, bty = "n")
  if (color_by == "XiLabel")
    rgl::legend3d("topright",
                  legend = c("0 Grass/remaining", "1 Stem", "2 Branch",
                             "3 Foliage", "4 Downed log",
                             "5 Sapling stem", "6 Below-canopy branch",
                             "7 Ground"),
                  col    = c("#aaff01", "#f02b00", "#7a4101",
                             "#0a9e00", "#9f0aef",
                             "#f54b8c", "#ae5504", "#0000fe"),
                  pch = 19, bty = "n", cex = 0.7)
  invisible(ds)
}

# forest_component_labels(): derive 8-class forest fuel component label per point.
# Uses: Classification (ground), TreeFilterLabel (overstory mask),
#       StemCls (stem detection), WoodLabel (wood/foliage), DownedLog (PCA tilt).
#
# Xi 2023 class definitions applied here:
#   Trees (overstory): stems/branches/foliage with height >= 5 m
#   Sapling stem (5): StemCls==2 in understory (TreeFilter class 1, height < 5 m)
#   Below-canopy branch (6): non-stem wood, diameter > 5 cm, understory layer
#   Downed log (4): wood near ground (Z <= downed_z_max) with primary PCA axis
#     tilted > downed_tilt_min_deg (default 60 deg) from vertical; the 60 deg
#     threshold clearly demarcates the leaning-tree layer from downed woody debris
#     (Xi 2023). Written to las@data$DownedLog by Section 4.
forest_component_labels <- function(ds) {
  if (inherits(ds, "LAS")) ds <- ds@data
  n   <- nrow(ds)
  lbl <- integer(n)   # default 0 = grass/remaining

  # Ground (LAS class 2)
  if ("Classification" %in% names(ds))
    lbl[ds$Classification == 2L] <- 7L
  not_ground <- lbl != 7L

  has_tf  <- "TreeFilterLabel" %in% names(ds)
  has_wc  <- "WoodLabel"       %in% names(ds)
  has_sc  <- "StemCls"         %in% names(ds)

  ov <- not_ground & if (has_tf) ds$TreeFilterLabel == 2L else rep(TRUE, n)
  us <- not_ground & if (has_tf) ds$TreeFilterLabel == 1L else rep(FALSE, n)

  # ---- Overstory ----
  # Foliage: overstory + WoodLabel == 1
  if (has_wc) lbl[ov & ds$WoodLabel == 1L] <- 3L

  # Branch: overstory + wood + not stem
  if (has_wc && has_sc)
    lbl[ov & ds$WoodLabel >= 2L & ds$StemCls != 2L] <- 2L
  else if (has_wc)
    lbl[ov & ds$WoodLabel >= 2L] <- 2L

  # Stem: overstory + StemCls == 2 (highest overstory priority)
  if (has_sc) lbl[ov & ds$StemCls == 2L] <- 1L

  # ---- Understory (TreeFilterLabel == 1; height < 5 m per Xi 2023) ----
  # Below-canopy foliage stays 0 (grass/remaining)
  # Below-canopy branch: understory + WoodLabel >= 2 + not sapling stem
  # (Xi 2023: non-stem wood with diameter > 5 cm in sapling/shrub/surface layers)
  if (has_wc && has_sc)
    lbl[us & ds$WoodLabel >= 2L & ds$StemCls != 2L] <- 6L
  else if (has_wc)
    lbl[us & ds$WoodLabel >= 2L] <- 6L

  # Sapling stem: understory + StemCls == 2; height < 5 m (highest understory priority)
  if (has_sc) lbl[us & ds$StemCls == 2L] <- 5L

  # ---- Downed log (class 4) -----------------------------------------------
  # Overrides understory branch (6) and understory stem (5) for wood points
  # flagged as horizontally-oriented by the PCA tilt test in Section 4.
  # Does NOT override ground (7) or overstory classes.
  has_dl <- "DownedLog" %in% names(ds)
  if (has_dl && has_wc)
    lbl[ds$DownedLog & ds$WoodLabel >= 2L] <- 4L

  lbl
}



# ============================================================================
# ============================================================================
# 1. READ
# ============================================================================

las <- readLAS(tls_input, select = "xyzicrnRGB")
if (is.empty(las)) stop("Input LAS is empty / failed to read.")
las_check(las)
las <- filter_duplicates(las)

# 2. PRE-PROCESS (full density)
# ============================================================================

# 2.1 Ground classification (CSF) + height normalization + range filter ------
las <- classify_ground(las, csf(
  sloop_smooth     = TRUE,
  class_threshold  = 0.05,
  cloth_resolution = 0.4,
  rigidness        = 3L,
  time_step        = 0.65
))

las <- normalize_height(las, tin())
las <- filter_poi(las, Z <= 50)   # m; hard ceiling after normalization

# 2.2 Statistical outlier removal --------------------------------------------
npts0 <- npoints(las)
las <- classify_noise(las, sor(k = 8, m = 3))
las <- filter_poi(las, Classification != LASNOISE)
message(sprintf("SOR: removed %d / %d points (%.2f%%) as noise.",
                npts0 - npoints(las), npts0,
                100 * (npts0 - npoints(las)) / npts0))

writeLAS(las, file.path(out_dir, "checkpoint_01_preproc.laz"))
# QC: plot_cloud_qc(las, color_by = "Classification")


# ============================================================================
# 3. SEGMENT
# ============================================================================

las <- treeAIBoxR::treeisonet(
  las,
  stemcls_model    = "treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)",
  treeloc_model    = "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)",
  crownoff_model   = "treeisonet_tls_boreal_crownoff_esegformer3D_128_15cm(GPU4GB)",
  treefilter_model = "treefiltering_tls_esegformer3D_128_8cm(GPU3GB)",
  cuda             = TRUE,
  verbose          = TRUE
)

# Trim: drop trees whose centroid falls outside (plot_radius + plot_buffer)
local({
  r_max    <- plot_radius + plot_buffer
  d        <- las@data[!is.na(TreeID) & TreeID > 0L,
                       .(cx = median(X), cy = median(Y)), by = TreeID]
  d$dist   <- sqrt((d$cx - plot_center_x)^2 + (d$cy - plot_center_y)^2)
  plot_ids <- d$TreeID[d$dist <= r_max]
  las     <<- filter_poi(las, TreeID %in% plot_ids | is.na(TreeID) | TreeID == 0L)
  message(sprintf("[3] Plot filter: %d trees within %.2f m of center.",
                  length(plot_ids), r_max))
})

writeLAS(las, file.path(out_dir, "checkpoint_02_segmented.laz"))
# QC: plot_cloud_qc(las, color_by = "TreeID")


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

# Derive Xi 2023 8-class forest component labels (used in Section 5 QC PNGs
# and Section 7 interactive view). Adds ComponentClass to las@data.
las <- treeAIBoxR::forest_components(las)

writeLAS(las, file.path(out_dir, "checkpoint_03_woodcls.laz"))
# QC: plot_cloud_qc(las, color_by = "XiLabel")


# ============================================================================
# 5. Per-tree QC PNGs
# ============================================================================
# Four-panel side views per TreeID written to <out_dir>/<tree_qc_subdir>/.
# Top row: RGB XZ / RGB YZ (or component colours if no RGB).
# Bottom row: forest_component_labels() colouring (8-class palette).
# Not all classes will appear per tree -- legend is dynamic.
# Runs in batch (builds qc_las locally; does not require Section 7 rgl).
if (TRUE) {   # set to FALSE to skip QC PNGs
  qc_dir <- file.path(out_dir, tree_qc_subdir)
  if (dir.exists(qc_dir))
    invisible(file.remove(list.files(qc_dir, pattern = "\\.png$", full.names = TRUE)))
  dir.create(qc_dir, showWarnings = FALSE, recursive = TRUE)
  qc_las <- lidR::filter_poi(las, !is.na(TreeID) & TreeID > 0L)
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
