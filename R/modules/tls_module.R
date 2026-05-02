# ============================================================================
# TLS MODULE: Terrestrial Laser Scanner Processing
# ============================================================================
# Pipeline:
#   1. Read + inspect cloud (lidR readLAS / las_check / field inventory).
#   2. Pre-process at full density: dedupe, ground (CSF), normalize, SOR.
#   3. Stem bases + CSP segmentation (Tao 2015 / Larysch 2025).
#      3.1 Eigen geometry (fine + optional coarse / multi-scale).
#      3.2 Sub-canopy foliage filter (global or stratified, optional gates).
#      3.3 Stem base detection (shape-aware DBSCAN, plot-extent + continuity).
#      3.4 CSP cost segmentation (assigns TreeID to every point).
#   4. Forest inventory: DBH, height, position, crown projection per tree.
#   5. Interactive plotting: 3D whole-plot + single-tree + 2D cross-sections.
#   6. Field-data validation: buffered candidate search + multi-criteria match.
#   7. TODO -- QSM (aRchi). Currently disabled while we hone segmentation.
#
# Refs:
#   https://r-lidar.github.io/lidRbook/gnd.html
#   https://cran.r-project.org/package=CspStandSegmentation
#   Tao et al. 2015 ISPRS J. Photogramm. Remote Sens. 110:66-76
#   Larysch et al. 2025 doi:10.1007/s10342-025-01796-z
# ============================================================================


# ---- Packages + parallelism ------------------------------------------------

library(lidR)
library(CspStandSegmentation)
library(aRchi)
library(ITSMe)
library(sf)

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

tls_params <- list(

  # --- Section 2: Ground + height filter + noise removal ------------------
  # CSF (Cloth Simulation Filter) chosen for dense low understory: with a
  # rigid cloth + tight class_threshold it follows true ground beneath
  # shrubs instead of latching onto shrub tops the way PTD seed selection does.
  csf_class_threshold  = 0.05,
  csf_cloth_resolution = 0.4,
  csf_rigidness        = 3L,
  csf_sloop_smooth     = TRUE,
  csf_time_step        = 0.65,
  z_min                = 0.2,    # m; height filter post-normalize
  z_max                = 50,
  # SOR: drop points whose mean k-NN distance is > m * sd above cloud-wide mean.
  sor_k                = 8,
  sor_m                = 3,      # lower = more aggressive

  # --- Section 3.1: Eigen geometry ----------------------------------------
  # add_geometry() runs at one neighborhood scale (fine). Multi-scale toggles
  # on a coarse pass via voxel-downsample + nearest-neighbor lookup back.
  eigen_n_cores        = parallel::detectCores(logical = TRUE),
  multiscale_enable    = FALSE,
  multiscale_voxel     = 0.10,   # m; coarse downsample voxel

  # --- Section 3.2: Sub-canopy foliage filter -----------------------------
  # Drops points classified as leaf-like (high Planarity) OR needle-like
  # (high Sphericity) AND not stem-like (low Linearity), only below
  # shrub_height_max. Two modes (toggle with shrub_strata_enable):
  #   * Global:     single rule from 0..shrub_height_max.
  #   * Stratified: per-stratum rules (ground/shrub/lower-canopy).
  # Optional additive gates (intensity, ExG, multi-scale) make the rule
  # strictly more conservative (drop only if all enabled signals agree).
  shrub_height_max     = 4.0,    # m; only filter below this height (HAG)
  shrub_z_min          = 0.10,   # m; do NOT filter below this height. Points
                                 # this close to ground have noisy eigen
                                 # features (neighborhood mixes trunk +
                                 # ground) and the ground classifier already
                                 # handles them.

  # Global rule (used when shrub_strata_enable = FALSE)
  shrub_planarity_min  = 0.4,
  shrub_sphericity_min = 0.35,
  shrub_linearity_max  = 0.4,

  # Stratified rule (used when shrub_strata_enable = TRUE)
  shrub_strata_enable    = TRUE,
  shrub_z_ground         = 0.5,   # ground stratum top  [0, 0.5]
  shrub_z_shrub          = 2.0,   # shrub stratum top   [0.5, 2.0]
                                  # lower stratum spans [2.0, shrub_height_max]
  shrub_g_planarity_min  = 0.30,  # ground: aggressive (CWD, litter)
  shrub_g_sphericity_min = 0.25,
  shrub_g_linearity_max  = 0.50,
  shrub_s_planarity_min  = 0.40,  # shrub: moderate (ferns, low brush)
  shrub_s_sphericity_min = 0.35,
  shrub_s_linearity_max  = 0.30,  # lowered 0.40 -> 0.30 to protect trunks
  shrub_l_planarity_min  = 0.55,  # lower: gentle (protect saplings)
  shrub_l_sphericity_min = 0.50,
  shrub_l_linearity_max  = 0.20,  # lowered 0.30 -> 0.20 to protect trunks

  # Additive signal gates (NA = disabled)
  shrub_intensity_max    = NA_real_,   # e.g. 8000: foliage tends to weak return
  shrub_exg_min          = NA_real_,   # e.g. 0.05: foliage is greener

  # --- Verticality-aware stem guard --------------------------------------
  # Linearity alone says "the local neighborhood is elongated" but not in
  # which direction. A horizontal twig / fallen branch has high Linearity
  # but is NOT a trunk. The guard re-defines "stem-like" as both
  # Linearity >= stem_lin_min AND Verticality >= stem_vert_min.
  # Per-stratum thresholds: ground stratum is loosest because eigen
  # features near the floor are noisy (neighborhood mixes trunk + ground),
  # so stem bottoms otherwise fail the guard and get dropped.
  stem_guard_enable      = TRUE,
  stem_g_lin_min         = 0.30,   # ground: very loose -- protect stem bottoms
  stem_g_vert_min        = 0.55,
  stem_s_lin_min         = 0.45,   # shrub: moderate
  stem_s_vert_min        = 0.75,
  stem_l_lin_min         = 0.55,   # lower-canopy: strict
  stem_l_vert_min        = 0.85,
  # Back-compat globals (used when shrub_strata_enable = FALSE)
  stem_lin_min           = 0.55,
  stem_vert_min          = 0.85,

  # QC plot toggle (rgl slab; red = dropped, green = kept)
  shrub_qc_plot          = TRUE,
  # --- Section 3.3: Stem base detection -----------------------------------
  # Shape-aware: keep slice points that are vertical AND planar (trunk-like),
  # then 3D DBSCAN-cluster them. 3D clustering separates leaning/close stems
  # that 2D density would merge.
  base_zmin            = 1.0,    # m; above shrub layer
  base_zmax            = 2.0,    # m; below most branching
  base_res             = 0.04,   # m; 3D DBSCAN eps -- tight enough to keep
                                 #    sapling clusters and close thin stems
                                 #    in separate clusters
  base_min_vert        = 0.85,
  base_min_plan        = 0.50,
  base_min_cluster     = 40,     # 40 captures saplings; 75 lost ~17 trees
                                 # on Plot 311 (vs field count of 53).
  base_merge_eps       = 0.10,   # m; collapse near-duplicate bases (DBSCAN
                                 # sometimes splits one trunk; without this,
                                 # CSP routes each duplicate's voxels to
                                 # whichever stem wins the Dijkstra tiebreak).
  # Raster fallback (uncomment + swap call below if geom misbehaves):
  # base_density_q       = 0.99,
  # base_merge_eps_raster= 0.15,

  # Plot extent (drop bases outside the buffered radius before CSP)
  plot_center_x        = 0.0,    # m; plot center (scanner-local frame)
  plot_center_y        = 0.0,
  plot_radius          = 11.28,  # m
  plot_buffer          = 1.72,   # m  (-> 13 m total)

  # Vertical-continuity validator (opt-in). For each base, count 1 m bands
  # above base_zmax with at least one point within base_continuity_radius
  # XY. Real trees produce a continuous column; shrub clusters that fooled
  # the base detector produce a stub.
  base_continuity_enable     = FALSE,
  base_continuity_radius     = 0.30,   # m
  base_continuity_top        = 6.0,    # m
  base_continuity_min_bands  = 4L,

  # --- Section 3.4: CSP segmentation --------------------------------------
  csp_voxel            = 0.07,   # m; routing graph voxel. 0.05 fragments
                                 # the graph after the shrub filter (igraph
                                 # assertion failure, ~75% base drop-out).
  csp_v_w              = 0.3,    # verticality weight (low: don't penalize
                                 # leaning trunks)
  csp_l_w              = 0.3,    # linearity weight  (mild branch penalty)
  csp_s_w              = 0.7,    # sphericity weight (strong foliage penalty)
  csp_n_cores          = 6L,     # CSP forks the full LAS + per-worker graph;
                                 # 24 cores OOMs Dijkstra on dense plots.
                                 # 6-8 is the sweet spot.

  # --- Section 4: Forest inventory + tree filter --------------------------
  inv_slice_min        = 0.3,    # m; bottom of taper band
  inv_slice_max        = 4.0,    # m; top of taper band
  inv_increment        = 0.1,    # m; vertical step
  inv_width            = 0.05,   # m; slice thickness for circle fit
  inv_max_dbh          = 1.0,    # m; reject impossible stems
  inv_n_cores          = parallel::detectCores(logical = TRUE),
  tree_min_dbh         = 0.05,   # m; >=5 cm DBH to count as a tree
  tree_min_height      = 5.0,    # m; >=5 m tall to count as a tree

  # Stem-base XY recomputation. forest_inventory() returns X/Y at breast
  # height (~1.3 m, splined from the taper). For leaning trunks that's not
  # where the field crew measured (base of stem). After the inventory, we
  # take each TreeID's points in [base_slice_zmin, base_slice_zmax] and
  # compute median XY -> stored as inv$X_base / inv$Y_base. Polar coords
  # and field matching use these base coords by default; the original BH
  # values are preserved as inv$X_bh / inv$Y_bh.
  base_slice_zmin      = 0.1,    # m; just above ground
  base_slice_zmax      = 0.5,    # m; below shrub interaction zone
  base_min_points      = 20L,    # min points to trust median; else fallback to BH XY
  match_use_base       = TRUE,   # FALSE -> match on BH X/Y (legacy)

  # --- Voxel sizes (driven by scanner accuracy) ---------------------------
  # SLAM report shows 100% of points within 5 cm of true surfaces, so any
  # spacing finer than this is geometrically redundant.
  tls_accuracy_m       = 0.05,
  # inv_thin_voxel: 1/5th of accuracy keeps fine taper detail and keeps
  # per-slice point counts under ~40k (forest_inventory has a quantile
  # bug at higher counts, patched below).
  # qsm_voxel: 2/5th of accuracy = densest spacing that still adds info.
  # Both derived after CONFIG closes.

  # --- Section 6.2: Leaf-wood separation ----------------------------------
  # Eigen-feature based; eigen columns already exist on the cloud from
  # Section 3.1. Wood = high Linearity AND low Sphericity AND low Planarity.
  # On conifers, sphericity does most of the work; on broadleaves (At),
  # planarity is the dominant gate.
  lws_linearity_min    = 0.40,
  lws_sphericity_max   = 0.30,
  lws_planarity_max    = 0.65,

  # Optional non-geometric gates (NA = disabled):
  #   Intensity: bark reflects more strongly + uniformly than foliage.
  #   HAG      : forces points above this absolute height to be foliage
  #              (catches branch-tip wood FPs in upper crown).
  #   ExG      : Excess Green = (2G - R - B) / max(RGB). Foliage is greener.
  lws_intensity_min    = 2500,
  lws_hag_max_for_wood = NA_real_,
  lws_exg_max          = 0.10,

  # --- Section 6: Field-data validation -----------------------------------
  # For each field tree, find ALL TLS candidates inside a buffer radius,
  # then pick the candidate with the lowest weighted score across
  # available metrics (xy / dbh / height / azimuth / distance).
  # Matched field info is written back as columns on `inv` (prefix `f_`).
  # Setting field_xlsx_path = NA disables Section 6.
  field_xlsx_path      = "C:/Users/AGRIST/OneDrive - Government of BC/Sklar, Daniel FOR_EX's files - Great Beaver Lake/trees.xlsx",
  field_sheet          = 1,
  field_col_id         = "tree_id",
  field_col_x          = NA_character_,    # NA -> derive from az + dist
  field_col_y          = NA_character_,
  field_col_az         = "az_deg",         # field azimuth (deg, 0 = +Y / N)
  field_col_dist       = "hd_m",           # field horizontal distance (m)
  field_col_dbh        = "2022_DBH",       # cm in xlsx (see field_dbh_units)
  field_col_height     = "2022_HT",        # m
  field_col_plot       = "Plot",           # NA disables plot filter
  field_plot_filter    = 311,              # value in field_col_plot to keep
  field_dbh_units      = "cm",             # "cm" or "m"; converts to m to match inv$DBH
  field_height_units   = "m",

  field_match_buffer   = 1.5,    # m; radius around each field tree to
                                 # gather TLS candidates from `inv`
  field_match_buffer_pass1 = 3.0,# m; wider buffer for the first pass so a
                                 # rotated plot still yields candidates for
                                 # the azimuth-bias estimate
  # Score weights (set 0 to ignore a metric; missing metrics auto-skip).
  # Normalization sigmas express "1 unit of cost" for each metric; tune
  # so that 1 m of XY error feels comparable to 5 cm DBH or 2 m height.
  field_w_xy           = 1.0,    field_sigma_xy   = 1.0,    # m
  field_w_dbh          = 1.0,    field_sigma_dbh  = 0.05,   # m
  field_w_height       = 0.5,    field_sigma_h    = 2.0,    # m
  field_w_az           = 0.3,    field_sigma_az   = 10.0,   # deg
  field_w_dist         = 0.5,    field_sigma_dist = 1.0,    # m

  # Two-pass match with auto azimuth-bias correction. After the first
  # match, estimate the median rotation between TLS and field bearings,
  # rotate inv$azi/X/Y by that amount, and re-match with a tighter buffer.
  field_az_bias_correct = TRUE,
  field_az_bias_min_n   = 10,    # min matches required to trust the bias
  field_az_bias_min_deg = 0.5,   # ignore biases smaller than this (no-op)
  field_az_bias_max_deg = 30     # safety cap; ignore biases larger than this
)

# Derived voxel sizes -- never exceed scanner accuracy.
tls_params$inv_thin_voxel <- min(0.01, tls_params$tls_accuracy_m / 5)
tls_params$qsm_voxel      <- min(0.02, tls_params$tls_accuracy_m * 0.4)


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
  if (!is.null(title)) rgl::title3d(main = title, col = "black")
  if (!is.null(mask))
    rgl::legend3d("topright",
                  legend = c("target (mask=TRUE)", "rest"),
                  col = c("#D62728", "#2CA02C"), pch = 19, bty = "n")
  invisible(ds)
}


# ============================================================================
# 1. READ + INSPECT
# ============================================================================

las <- readLAS(tls_input, select = "xyzicrnRGB")  # keep core attrs + RGB
if (is.empty(las)) stop("Input LAS is empty / failed to read.")

las_check(las)
print(las)

# LAS field inventory -- which extra signals are actually populated?
# Used to decide whether the optional Intensity / NumberOfReturns / RGB
# gates in the leaf-wood separation block (Section 6) are worth enabling.
local({
  d <- las@data
  message("LAS field inventory:")
  message("  columns: ", paste(names(d), collapse = ", "))
  if ("Intensity" %in% names(d)) {
    qi <- stats::quantile(d$Intensity, c(0.05, 0.5, 0.95), na.rm = TRUE)
    message(sprintf("  Intensity: range [%g, %g], q05/med/q95 = %g / %g / %g",
                    min(d$Intensity, na.rm = TRUE),
                    max(d$Intensity, na.rm = TRUE),
                    qi[1], qi[2], qi[3]))
  }
  if ("NumberOfReturns" %in% names(d)) {
    tab <- table(d$NumberOfReturns)
    message("  NumberOfReturns distribution: ",
            paste(sprintf("%s=%d", names(tab), as.integer(tab)),
                  collapse = ", "))
    if (length(tab) == 1L)
      message("    -> single-return only; return-number gate not useful here.")
  }
  rgb_cols <- intersect(c("R", "G", "B"), names(d))
  if (length(rgb_cols) == 3L) {
    rng <- range(c(d$R, d$G, d$B), na.rm = TRUE)
    if (rng[2] == 0) {
      message("  RGB present but all zero -- not usable.")
    } else {
      message(sprintf("  RGB: range [%g, %g], median R/G/B = %g / %g / %g",
                      rng[1], rng[2],
                      stats::median(d$R), stats::median(d$G),
                      stats::median(d$B)))
    }
  } else {
    message("  RGB: not present.")
  }
})

las <- filter_duplicates(las)


# ============================================================================
# 2. PRE-PROCESS (full density)
# ============================================================================

# 2.1 Ground classification (CSF) + height normalization + range filter ------
las <- classify_ground(las, csf(
  sloop_smooth     = tls_params$csf_sloop_smooth,
  class_threshold  = tls_params$csf_class_threshold,
  cloth_resolution = tls_params$csf_cloth_resolution,
  rigidness        = tls_params$csf_rigidness,
  time_step        = tls_params$csf_time_step
))

las <- normalize_height(las, tin())
las <- filter_poi(las, Z >= tls_params$z_min, Z <= tls_params$z_max)

# 2.2 Statistical outlier removal --------------------------------------------
# Drop scanner ghost / mixed-pixel noise. add_geometry() and density-raster
# base detection are both sensitive to isolated outliers; SOR removes them.
npts0 <- npoints(las)
las <- classify_noise(las, sor(k = tls_params$sor_k, m = tls_params$sor_m))
las <- filter_poi(las, Classification != LASNOISE)
message(sprintf("SOR: removed %d / %d points (%.2f%%) as noise.",
                npts0 - npoints(las), npts0,
                100 * (npts0 - npoints(las)) / npts0))


# ============================================================================
# 3. STEM BASES + CSP SEGMENTATION (Tao 2015)
# ============================================================================
# CSP builds a connectivity graph over the voxelized cloud, computes weighted
# shortest paths from every voxel to every detected stem base, and assigns
# each point to its graph-nearest stem. Path costs are weighted by eigen
# geometry (verticality, linearity, sphericity) so routing prefers stem-like
# structures and avoids leaking through foliage into neighbor crowns.

# 3.1 Eigen geometry ---------------------------------------------------------
# Fine-scale features required by base detection + CSP cost function.
las <- add_geometry(las, n_cores = tls_params$eigen_n_cores)

# Optional coarse-scale eigen (multi-scale consensus). Voxel-downsample,
# recompute eigen, then assign each full-density point the eigen of its
# nearest coarse point. The shrub filter (3.2) can require foliage at
# BOTH scales -- kills speckle from per-point neighborhood noise.
if (isTRUE(tls_params$multiscale_enable)) {
  npts0 <- npoints(las)
  message(sprintf("Multi-scale eigen: downsampling to %.2f m for coarse pass...",
                  tls_params$multiscale_voxel))
  coarse <- decimate_points(las,
              random_per_voxel(tls_params$multiscale_voxel, n = 1L))
  coarse <- add_geometry(coarse, n_cores = tls_params$eigen_n_cores)
  if (!requireNamespace("RANN", quietly = TRUE))
    stop("multiscale_enable=TRUE requires the RANN package.")
  nn <- RANN::nn2(coarse@data[, .(X, Y, Z)],
                  las@data[,    .(X, Y, Z)], k = 1)$nn.idx[, 1]
  data.table::setDT(las@data)
  las@data[, `:=`(
    Linearity_c  = coarse@data$Linearity[nn],
    Planarity_c  = coarse@data$Planarity[nn],
    Sphericity_c = coarse@data$Sphericity[nn]
  )]
  message(sprintf("Multi-scale eigen: attached coarse features to %d points.", npts0))
  rm(coarse, nn); invisible(gc())
}

# 3.2 Sub-canopy foliage filter ----------------------------------------------
# Drops leaf-like (Planarity high) OR needle-like (Sphericity high) AND
# not-stem-like (Linearity low) points below shrub_height_max.
# Modes: global (single rule) or stratified (ground/shrub/lower).
# Optional additive gates: multi-scale consensus, intensity, ExG.
# All gates are conservative -- they can only KEEP more points.
npts0 <- npoints(las)
shrub_layer <- las@data$Z >= tls_params$shrub_z_min &
               las@data$Z <  tls_params$shrub_height_max

# (a) Geometric foliage rule (fine scale)
# stem_like requires BOTH high Linearity AND high Verticality so that
# horizontal twigs / fallen branches (high Linearity, low Verticality) are
# NOT protected. Falls back to the old Linearity-only guard if either
# Verticality is missing or the user disables stem_guard_enable.
has_vert <- "Verticality" %in% names(las@data)
use_stem_guard <- isTRUE(tls_params$stem_guard_enable) && has_vert
if (isTRUE(tls_params$shrub_strata_enable)) {
  z <- las@data$Z
  # Per-point stratum: 1 = ground, 2 = shrub, 3 = lower-canopy.
  st <- ifelse(z <  tls_params$shrub_z_ground, 1L,
        ifelse(z <  tls_params$shrub_z_shrub,  2L, 3L))
  pmin_v <- c(tls_params$shrub_g_planarity_min,
              tls_params$shrub_s_planarity_min,
              tls_params$shrub_l_planarity_min)[st]
  smin_v <- c(tls_params$shrub_g_sphericity_min,
              tls_params$shrub_s_sphericity_min,
              tls_params$shrub_l_sphericity_min)[st]
  lmax_v <- c(tls_params$shrub_g_linearity_max,
              tls_params$shrub_s_linearity_max,
              tls_params$shrub_l_linearity_max)[st]
  foliage_blob <- (las@data$Planarity  >= pmin_v |
                   las@data$Sphericity >= smin_v)
  if (use_stem_guard) {
    # Per-stratum stem-guard thresholds (looser near the ground).
    slin_v <- c(tls_params$stem_g_lin_min,
                tls_params$stem_s_lin_min,
                tls_params$stem_l_lin_min)[st]
    svrt_v <- c(tls_params$stem_g_vert_min,
                tls_params$stem_s_vert_min,
                tls_params$stem_l_vert_min)[st]
    stem_like <- las@data$Linearity   >= slin_v &
                 las@data$Verticality >= svrt_v
    foliage_shape <- foliage_blob & !stem_like
    rm(stem_like, slin_v, svrt_v)
  } else {
    foliage_shape <- foliage_blob & las@data$Linearity <= lmax_v
  }
  rm(z, st, pmin_v, smin_v, lmax_v, foliage_blob)
} else {
  foliage_blob <- (las@data$Planarity  >= tls_params$shrub_planarity_min |
                   las@data$Sphericity >= tls_params$shrub_sphericity_min)
  if (use_stem_guard) {
    stem_like <- las@data$Linearity   >= tls_params$stem_lin_min &
                 las@data$Verticality >= tls_params$stem_vert_min
    foliage_shape <- foliage_blob & !stem_like
    rm(stem_like)
  } else {
    foliage_shape <- foliage_blob & las@data$Linearity <= tls_params$shrub_linearity_max
  }
  rm(foliage_blob)
}

# (b) Multi-scale consensus: also require foliage at the coarse scale.
if (isTRUE(tls_params$multiscale_enable) &&
    all(c("Linearity_c", "Planarity_c", "Sphericity_c") %in% names(las@data))) {
  has_vert_c <- "Verticality_c" %in% names(las@data)
  use_stem_guard_c <- isTRUE(tls_params$stem_guard_enable) && has_vert_c
  if (isTRUE(tls_params$shrub_strata_enable)) {
    # Reuse the same per-stratum thresholds at coarse scale.
    z <- las@data$Z
    st <- ifelse(z <  tls_params$shrub_z_ground, 1L,
          ifelse(z <  tls_params$shrub_z_shrub,  2L, 3L))
    pmin_v <- c(tls_params$shrub_g_planarity_min,
                tls_params$shrub_s_planarity_min,
                tls_params$shrub_l_planarity_min)[st]
    smin_v <- c(tls_params$shrub_g_sphericity_min,
                tls_params$shrub_s_sphericity_min,
                tls_params$shrub_l_sphericity_min)[st]
    lmax_v <- c(tls_params$shrub_g_linearity_max,
                tls_params$shrub_s_linearity_max,
                tls_params$shrub_l_linearity_max)[st]
    foliage_coarse <- (las@data$Planarity_c  >= pmin_v |
                       las@data$Sphericity_c >= smin_v) &
                       las@data$Linearity_c  <= lmax_v
    rm(z, st, pmin_v, smin_v, lmax_v)
  } else {
    foliage_coarse <- (las@data$Planarity_c  >= tls_params$shrub_planarity_min |
                       las@data$Sphericity_c >= tls_params$shrub_sphericity_min) &
                       las@data$Linearity_c  <= tls_params$shrub_linearity_max
  }
  foliage_shape <- foliage_shape & foliage_coarse
  rm(foliage_coarse)
}

# (c) Additive signal gates (intensity / ExG). Drop only if every enabled
#     signal also calls the point foliage.
weak_signal <- rep(TRUE, npoints(las))
if (!is.na(tls_params$shrub_intensity_max) && "Intensity" %in% names(las@data))
  weak_signal <- weak_signal & las@data$Intensity <= tls_params$shrub_intensity_max
if (!is.na(tls_params$shrub_exg_min) && all(c("R", "G", "B") %in% names(las@data))) {
  exg_full <- (2 * as.numeric(las@data$G) -
                   as.numeric(las@data$R) - as.numeric(las@data$B)) /
              max(c(las@data$R, las@data$G, las@data$B), na.rm = TRUE)
  weak_signal <- weak_signal & exg_full >= tls_params$shrub_exg_min
  rm(exg_full)
}

# Combine the three gates into the base drop mask.
drop_mask <- shrub_layer & foliage_shape & weak_signal

# (d) Apply filter (snapshotting coords first if QC plot is enabled)
if (isTRUE(tls_params$shrub_qc_plot)) {
  pre_x <- las@data$X; pre_y <- las@data$Y; pre_z <- las@data$Z
}

las <- filter_poi(las, !drop_mask)
message(sprintf("Shrub filter [%s]: removed %d / %d points (%.2f%%) below %.1f m.",
                if (isTRUE(tls_params$shrub_strata_enable)) "stratified" else "global",
                npts0 - npoints(las), npts0,
                100 * (npts0 - npoints(las)) / npts0,
                tls_params$shrub_height_max))

# (e) Optional QC plot: full-cloud rgl. Red = dropped, green = kept.
#     Toggle via tls_params$shrub_qc_plot.
if (isTRUE(tls_params$shrub_qc_plot)) {
  pre_dt <- data.frame(X = pre_x, Y = pre_y, Z = pre_z)
  plot_cloud_qc(
    pre_dt,
    mask  = drop_mask,
    title = sprintf("Shrub filter QC (red=dropped, green=kept) | %d / %d dropped (%.1f%%)",
                    sum(drop_mask), length(drop_mask),
                    100 * mean(drop_mask))
  )
  rm(pre_dt)
}
rm(shrub_layer, foliage_shape, weak_signal, drop_mask)
if (exists("pre_x")) rm(pre_x, pre_y, pre_z)

# 3.3 Stem base detection ----------------------------------------------------
# Shape-aware base finder. Local fast variant of
# CspStandSegmentation::find_base_coordinates_geom() that reuses the
# eigen columns we already computed in 3.1. Otherwise identical logic
# (CspStandSegmentation 0.2.0).
find_bases_geom_fast <- function(las, zmin, zmax, res,
                                 min_verticality, min_planarity,
                                 min_cluster_size, merge_eps = 0.20) {
  slice <- lidR::filter_poi(las, Classification != 2L & Z > zmin & Z < zmax)
  if (lidR::is.empty(slice))
    stop("No points in [zmin, zmax].")
  if (!all(c("Verticality", "Planarity") %in% names(slice@data)))
    slice <- CspStandSegmentation::add_geometry(slice)
  slice <- lidR::filter_poi(slice,
                            Planarity   > min_planarity &
                            Verticality > min_verticality)
  if (lidR::is.empty(slice))
    stop("No points pass planarity/verticality thresholds.")
  cl <- dbscan::dbscan(slice@data[, 1:3], eps = res, minPts = 1)$cluster
  keep <- as.integer(names(table(cl))[table(cl) > min_cluster_size])
  slice <- lidR::filter_poi(slice, cl %in% keep)
  cl    <- cl[cl %in% keep]
  xy <- aggregate(slice@data[, 1:2], by = list(cl), mean)
  z  <- aggregate(slice@data[, 3],   by = list(cl), min)
  bases <- data.frame(X = xy[, 2], Y = xy[, 3], Z = z[, 2])

  # Merge near-duplicate bases (DBSCAN can split one trunk into multiple
  # vertical-planar fragments at slightly different heights). Collapse any
  # bases within `merge_eps` (XY) into a single mean centroid. Without
  # this step, CSP routes each duplicate seed's voxels to whichever stem
  # wins the Dijkstra tiebreak, absorbing entire neighbor trunks.
  if (nrow(bases) > 1 && merge_eps > 0) {
    mc <- dbscan::dbscan(bases[, c("X", "Y")], eps = merge_eps, minPts = 1)$cluster
    bases <- aggregate(bases, by = list(mc), mean)[, -1]
  }
  bases$TreeID <- seq_len(nrow(bases))
  bases
}

bases <- find_bases_geom_fast(las,
  zmin             = tls_params$base_zmin,
  zmax             = tls_params$base_zmax,
  res              = tls_params$base_res,
  min_verticality  = tls_params$base_min_vert,
  min_planarity    = tls_params$base_min_plan,
  min_cluster_size = tls_params$base_min_cluster,
  merge_eps        = tls_params$base_merge_eps
)
# Upstream alternatives (uncomment + swap if needed):
# bases <- find_base_coordinates_geom(las, ...)        # slower, identical result
# bases <- find_base_coordinates_raster(las, ...)      # faster, merges close stems

# Plot-extent filter: drop bases outside the buffered radius BEFORE CSP --
# avoids routing the entire cloud to seeds we'll discard later.
nb0 <- nrow(bases)
base_dist <- sqrt((bases$X - tls_params$plot_center_x)^2 +
                  (bases$Y - tls_params$plot_center_y)^2)
bases <- bases[base_dist <= (tls_params$plot_radius + tls_params$plot_buffer), ]
message(sprintf("Plot-extent base filter: kept %d / %d bases inside %.2f m + %.2f m buffer.",
                nrow(bases), nb0, tls_params$plot_radius, tls_params$plot_buffer))

# Vertical-continuity validator (opt-in). Drop bases that lack a tall
# continuous column above them -- they're shrub clusters that fooled
# the base detector, not real trees.
if (isTRUE(tls_params$base_continuity_enable) && nrow(bases) > 0) {
  if (!requireNamespace("RANN", quietly = TRUE))
    stop("base_continuity_enable=TRUE requires the RANN package.")
  nb1 <- nrow(bases)
  bands <- seq(tls_params$base_zmax,
               tls_params$base_continuity_top, by = 1.0)
  if (length(bands) < 2L)
    stop("base_continuity_top must be > base_zmax + 1.0 m.")
  # XY-only KD-tree of full cloud; bands compared via Z lookup per query.
  cloud_xyz <- as.matrix(las@data[, .(X, Y, Z)])
  bands_per_base <- vapply(seq_len(nrow(bases)), function(i) {
    bx <- bases$X[i]; by <- bases$Y[i]
    dxy2 <- (cloud_xyz[, 1] - bx)^2 + (cloud_xyz[, 2] - by)^2
    near <- dxy2 <= tls_params$base_continuity_radius^2
    if (!any(near)) return(0L)
    z_near <- cloud_xyz[near, 3]
    sum(vapply(seq_len(length(bands) - 1L), function(k)
      any(z_near >= bands[k] & z_near < bands[k + 1L]),
      logical(1)))
  }, integer(1))
  bases <- bases[bands_per_base >= tls_params$base_continuity_min_bands, ]
  message(sprintf("Vertical-continuity filter: kept %d / %d bases (>=%d of %d bands).",
                  nrow(bases), nb1,
                  tls_params$base_continuity_min_bands,
                  length(bands) - 1L))
  rm(cloud_xyz, bands_per_base)
}

# 3.4 CSP cost segmentation --------------------------------------------------
# Assigns TreeID to every point.
las <- csp_cost_segmentation(las, bases,
  Voxel_size = tls_params$csp_voxel,
  V_w        = tls_params$csp_v_w,
  L_w        = tls_params$csp_l_w,
  S_w        = tls_params$csp_s_w,
  N_cores    = tls_params$csp_n_cores
)


# ============================================================================
# 4. FOREST INVENTORY
# ============================================================================
# Fits circles in thin slices and splines DBH/X/Y vs Z. Returns one row per
# TreeID with X, Y, DBH, Height, ConvexHullArea, quality_flag.

# 4.1 Patch CspStandSegmentation::forest_inventory ---------------------------
# Bug: q = 1 - sqrt(100 / nrow(slice)) + 0.05 exceeds 1 when slice has
# >~38k points (-> quantile() probs out of [0,1]). PR submitted upstream.
# Safe to delete this block once the merged version is reinstalled via:
#   remotes::install_github("JulFrey/CspStandSegmentation")
local({
  ns <- asNamespace("CspStandSegmentation")
  fi <- get("forest_inventory", envir = ns)
  src <- deparse(fi, control = c("useSource", "keepInteger", "keepNA"))
  patched <- sub(
    "q <- 1 - sqrt\\(100\\s*/\\s*nrow\\(slice\\)\\) \\+ 0.05",
    "q <- min(0.99, 1 - sqrt(100 / nrow(slice)) + 0.05)",
    src
  )
  if (identical(patched, src)) {
    warning("forest_inventory monkey-patch: target line not found ",
            "(upstream may have fixed it -- safe to remove this block).")
  } else {
    fi_new <- eval(parse(text = patched))
    environment(fi_new) <- ns
    assignInNamespace("forest_inventory", fi_new, ns = "CspStandSegmentation")
    message("Patched CspStandSegmentation::forest_inventory (quantile clamp at 0.99).")
  }
})


# 4.2 Run inventory + DBH/Height filter --------------------------------------
inv_input <- decimate_points(las, random_per_voxel(tls_params$inv_thin_voxel, n = 1L))

inv <- CspStandSegmentation::forest_inventory(inv_input,
  slice_min = tls_params$inv_slice_min,
  slice_max = tls_params$inv_slice_max,
  increment = tls_params$inv_increment,
  width     = tls_params$inv_width,
  max_dbh   = tls_params$inv_max_dbh,
  n_cores   = tls_params$inv_n_cores
)

rm(inv_input); gc()

# Drop shrubs / saplings / branch fragments -- keep only valid trees.
# (Plot-extent filter was already applied to bases pre-CSP.)
inv <- inv[!is.na(inv$DBH) &
           inv$DBH    >= tls_params$tree_min_dbh &
           inv$Height >= tls_params$tree_min_height, ]

message(sprintf("Inventory: %d trees retained after DBH/Height filter.", nrow(inv)))

# Stem-base XY: forest_inventory's X/Y is at breast height (1.3 m, splined).
# Recompute median XY in [base_slice_zmin, base_slice_zmax] from the
# segmented cloud so leaning trees' positions match field (base) measurements.
local({
  d <- las@data
  if (!"TreeID" %in% names(d)) {
    warning("TreeID missing from segmented cloud; skipping X_base/Y_base.")
    return()
  }
  z1 <- tls_params$base_slice_zmin
  z2 <- tls_params$base_slice_zmax
  base_dt <- d[!is.na(TreeID) & Z >= z1 & Z <= z2,
               .(X_base = median(X), Y_base = median(Y), n = .N),
               by = TreeID]
  m <- match(inv$TreeID, base_dt$TreeID)
  inv$X_bh   <<- inv$X            # preserve BH XY
  inv$Y_bh   <<- inv$Y
  inv$X_base <<- base_dt$X_base[m]
  inv$Y_base <<- base_dt$Y_base[m]
  inv$n_base <<- base_dt$n[m]
  # Fallback to BH XY where base slice is sparse / missing.
  bad <- is.na(inv$X_base) | is.na(inv$n_base) |
         inv$n_base < tls_params$base_min_points
  inv$X_base[bad] <<- inv$X_bh[bad]
  inv$Y_base[bad] <<- inv$Y_bh[bad]
  message(sprintf("Base XY: %d / %d trees from slice [%.2f, %.2f] m, %d fell back to BH.",
                  sum(!bad), nrow(inv), z1, z2, sum(bad)))
  # Promote base XY to inv$X / inv$Y so all downstream code uses base.
  inv$X <<- inv$X_base
  inv$Y <<- inv$Y_base
})

# Polar coordinates (azimuth + horizontal distance) for cruise comparison.
inv$azi        <- (atan2(inv$X, inv$Y) * 180 / pi) %% 360
inv$distance_m <- sqrt(inv$X^2 + inv$Y^2)

write.csv(inv, file.path(out_dir, "inventory.csv"), row.names = FALSE)
writeLAS(las,  file.path(out_dir, "plot_classified.laz"))


# ============================================================================
# 5. INTERACTIVE PLOTTING
# ============================================================================
# Whole-plot view (every TreeID labeled) + single-tree view at full density.

if (interactive()) {

  # Helper: draw L-shaped ground axes + Z axis around the active rgl scene.
  # offset: c(dx, dy) lidR shift; pts: data.table with X, Y; z_top: Z reach (m);
  # step_xy: grid spacing; hide_zero: blank "0" labels on X/Y (Z keeps it).
  draw_axes <- function(offset, pts, z_top, step_xy) {
    xr <- range(pts$X) - offset[1];  yr <- range(pts$Y) - offset[2]
    rng <- function(r, s) c(floor(r[1] / s) * s, ceiling(r[2] / s) * s)
    xb <- rng(xr, step_xy);  yb <- rng(yr, step_xy)
    rgl::points3d(c(xb, 0, 0), c(yb, 0, 0), c(0, 0, 0, 0, z_top), alpha = 0)
    rgl::axis3d("z--", at = seq(0, z_top, by = 5), col = "white")
    for (ax in c("x--", "y--")) {
      at <- seq(if (ax == "x--") xb[1] else yb[1],
                if (ax == "x--") xb[2] else yb[2], by = step_xy)
      rgl::axis3d(ax, at = at, labels = ifelse(at == 0, "", at), col = "white")
    }
  }

  # Drop unsegmented points and failed-filter trees; recolor per TreeID.
  las_seg <- filter_poi(las, TreeID %in% inv$TreeID)
  data.table::setDT(las_seg@data)
  if (all(c("R", "G", "B") %in% names(las_seg@data)))
    las_seg@data[, c("R", "G", "B") := NULL]
  # color_ids() needs n_neighbors < number of trees (RANN::nn2 fails
  # otherwise). Clamp to min(10, n_trees - 1) and bail out gracefully if
  # there are fewer than 2 trees.
  n_trees <- length(unique(inv$TreeID))
  if (n_trees < 2) {
    warning("color_ids skipped: need at least 2 trees, found ", n_trees)
  } else {
    las_seg <- color_ids(las_seg, n_neighbors = min(10L, n_trees - 1L))
  }

  # Whole plot
  x <- lidR::plot(las_seg, color = "RGB", axis = FALSE)
  rgl::texts3d(inv$X - x[1], inv$Y - x[2], inv$Height, text = inv$TreeID,
               col = "white", cex = 1.2, family = "mono", font = 2)
  draw_axes(x, las_seg@data,
            ceiling(max(inv$Height, na.rm = TRUE) / 5) * 5, step_xy = 5)

  # Single tree -- pick by the label you see in the plot.
  tid <- 77
  tree <- las_seg;  tree@data <- las_seg@data[TreeID == tid]
  y <- lidR::plot(tree, color = "RGB", size = 2, axis = FALSE)
  draw_axes(y, tree@data,
            ceiling(inv$Height[inv$TreeID == tid] / 5) * 5, step_xy = 1)

  # 5b. Full-plot QC -------------------------------------------------------
  # rgl view of the entire segmented cloud. For mask-based QC overlays:
  #   plot_cloud_qc(las, mask = drop_mask)
  plot_cloud_qc(las, color_by = "TreeID",
                title = "Segmented cloud (color = TreeID)")
}


# ============================================================================
# 6. FIELD-DATA VALIDATION
# ============================================================================
# For each field-tally tree:
#   1. Gather all TLS candidates within `field_match_buffer` (XY radius).
#   2. Score each candidate by weighted normalized residuals across
#      available metrics: XY distance, |delta_DBH|, |delta_Height|,
#      |delta_azimuth| (circular), |delta_distance|.
#   3. Pick the lowest-score candidate (one TLS tree may not be claimed twice).
# Matched field metadata is written back onto `inv` as `f_*` columns
# (f_id, f_dbh, f_height, f_az, f_dist, f_x, f_y, f_score, f_xy_residual).
# Skipped if field_xlsx_path is NA. Field xlsx must contain at minimum
# either (X, Y) or (azimuth, distance) plus DBH and Height.

if (!is.na(tls_params$field_xlsx_path) &&
    nzchar(tls_params$field_xlsx_path) &&
    file.exists(tls_params$field_xlsx_path)) {

  if (!requireNamespace("readxl", quietly = TRUE))
    stop("Section 6 requires the readxl package.")

  field <- as.data.frame(readxl::read_excel(
    tls_params$field_xlsx_path, sheet = tls_params$field_sheet))

  # Optional plot filter (xlsx may contain multiple plots).
  if (!is.na(tls_params$field_col_plot) &&
      nzchar(tls_params$field_col_plot) &&
      tls_params$field_col_plot %in% names(field) &&
      !is.null(tls_params$field_plot_filter) &&
      !is.na(tls_params$field_plot_filter)) {
    keep <- as.character(field[[tls_params$field_col_plot]]) ==
            as.character(tls_params$field_plot_filter)
    field <- field[!is.na(keep) & keep, , drop = FALSE]
    message(sprintf("Field xlsx: filtered to %s = %s -> %d rows.",
                    tls_params$field_col_plot,
                    as.character(tls_params$field_plot_filter), nrow(field)))
  }
  if (!nrow(field))
    stop("Section 6: no field rows after plot filter.")

  # Pull each configured column if its name is set + present in the sheet.
  # Always returns either NULL (column absent) or a vector of length nrow(field)
  # so data.frame() never sees a length-0 mismatch.
  pull <- function(col, type = as.numeric) {
    if (is.null(col) || length(col) != 1L || is.na(col) || !nzchar(col))
      return(NULL)
    if (!col %in% names(field)) {
      warning(sprintf("Field column '%s' not found in xlsx (have: %s).",
                      col, paste(names(field), collapse = ", ")))
      return(NULL)
    }
    v <- suppressWarnings(type(field[[col]]))
    if (length(v) != nrow(field)) {
      warning(sprintf("Field column '%s' coerced to length %d (expected %d); skipping.",
                      col, length(v), nrow(field)))
      return(NULL)
    }
    v
  }
  # Field ID column: fall back to row index if the configured name is missing.
  fid_col <- tls_params$field_col_id
  field_id_vec <- if (!is.null(fid_col) && length(fid_col) == 1L &&
                      !is.na(fid_col) && nzchar(fid_col) &&
                      fid_col %in% names(field)) {
    as.character(field[[fid_col]])
  } else {
    if (!is.null(fid_col) && !is.na(fid_col) && nzchar(fid_col))
      warning(sprintf("Field id column '%s' not found in xlsx; using row index.",
                      fid_col))
    as.character(seq_len(nrow(field)))
  }
  # Build fd column-by-column so any single bad pull() can't blow up the frame.
  fd <- data.frame(field_id = field_id_vec, stringsAsFactors = FALSE)
  add_col <- function(name, col_param) {
    v <- pull(col_param)
    if (is.null(v)) return(invisible(NULL))
    if (length(v) != nrow(fd)) {
      warning(sprintf("Field column '%s' (-> %s) length %d != %d; skipped.",
                      col_param, name, length(v), nrow(fd)))
      return(invisible(NULL))
    }
    fd[[name]] <<- v
    invisible(NULL)
  }
  add_col("field_x",      tls_params$field_col_x)
  add_col("field_y",      tls_params$field_col_y)
  add_col("field_az",     tls_params$field_col_az)
  add_col("field_dist",   tls_params$field_col_dist)
  add_col("field_dbh",    tls_params$field_col_dbh)
  add_col("field_height", tls_params$field_col_height)
  message(sprintf("Field columns built: %s",
                  paste(setdiff(names(fd), "field_id"), collapse = ", ")))

  # Unit conversions (DBH/Height -> meters to match inv$DBH, inv$Height).
  if (!is.null(fd$field_dbh) &&
      identical(tolower(tls_params$field_dbh_units), "cm"))
    fd$field_dbh <- fd$field_dbh / 100
  if (!is.null(fd$field_height) &&
      identical(tolower(tls_params$field_height_units), "cm"))
    fd$field_height <- fd$field_height / 100

  # Fill missing X/Y from azimuth + distance, or vice versa.
  if (is.null(fd$field_x) || all(is.na(fd$field_x))) {
    if (is.null(fd$field_az) || is.null(fd$field_dist))
      stop("Field data needs either (X, Y) or (azimuth, distance).")
    fd$field_x <- fd$field_dist * sin(fd$field_az * pi / 180) +
                  tls_params$plot_center_x
    fd$field_y <- fd$field_dist * cos(fd$field_az * pi / 180) +
                  tls_params$plot_center_y
  }
  if (is.null(fd$field_az))
    fd$field_az <- (atan2(fd$field_x - tls_params$plot_center_x,
                          fd$field_y - tls_params$plot_center_y) * 180 / pi) %% 360
  if (is.null(fd$field_dist))
    fd$field_dist <- sqrt((fd$field_x - tls_params$plot_center_x)^2 +
                          (fd$field_y - tls_params$plot_center_y)^2)
  fd <- fd[!is.na(fd$field_x) & !is.na(fd$field_y), ]

  # Circular azimuth distance (deg).
  az_diff <- function(a, b) abs(((a - b + 540) %% 360) - 180)

  # Compute candidate score for one (field row i, tls row j).
  score_pair <- function(i, j) {
    p   <- tls_params
    dx  <- inv$X[j]        - fd$field_x[i]
    dy  <- inv$Y[j]        - fd$field_y[i]
    sxy <- sqrt(dx^2 + dy^2)
    s   <- p$field_w_xy * (sxy / p$field_sigma_xy)
    if (!is.null(fd$field_dbh) && !is.na(fd$field_dbh[i]) && !is.na(inv$DBH[j]))
      s <- s + p$field_w_dbh *
             (abs(inv$DBH[j] - fd$field_dbh[i]) / p$field_sigma_dbh)
    if (!is.null(fd$field_height) && !is.na(fd$field_height[i]) && !is.na(inv$Height[j]))
      s <- s + p$field_w_height *
             (abs(inv$Height[j] - fd$field_height[i]) / p$field_sigma_h)
    if (!is.na(fd$field_az[i]))
      s <- s + p$field_w_az *
             (az_diff(inv$azi[j], fd$field_az[i]) / p$field_sigma_az)
    if (!is.na(fd$field_dist[i]))
      s <- s + p$field_w_dist *
             (abs(inv$distance_m[j] - fd$field_dist[i]) / p$field_sigma_dist)
    list(score = s, xy = sxy)
  }

  # Gather candidates within buffer for every field tree, then resolve
  # conflicts greedily: pick the global minimum score first, lock that
  # pair, repeat until no candidates remain. Wrapped in a function so we
  # can run the match twice -- once to estimate azimuth bias, once after
  # rotating inv by that bias.
  do_match <- function(buffer) {
    cands <- lapply(seq_len(nrow(fd)), function(i) {
      r <- sqrt((inv$X - fd$field_x[i])^2 + (inv$Y - fd$field_y[i])^2)
      j <- which(r <= buffer)
      if (!length(j)) return(NULL)
      sc <- vapply(j, function(jj) score_pair(i, jj)$score, numeric(1))
      data.frame(field_idx = i, tls_idx = j, score = sc, xy = r[j])
    })
    cand_tbl <- do.call(rbind, cands)

    # Reset f_* columns on `inv` (NA for unmatched TLS trees).
    inv$f_id          <<- NA_character_
    inv$f_x           <<- NA_real_
    inv$f_y           <<- NA_real_
    inv$f_az          <<- NA_real_
    inv$f_dist        <<- NA_real_
    inv$f_dbh         <<- NA_real_
    inv$f_height      <<- NA_real_
    inv$f_score       <<- NA_real_
    inv$f_xy_residual <<- NA_real_
    inv$f_delta_dbh    <<- NA_real_
    inv$f_delta_height <<- NA_real_
    inv$f_delta_az     <<- NA_real_
    inv$f_delta_dist   <<- NA_real_

    matched_field <- logical(nrow(fd))
    if (!is.null(cand_tbl) && nrow(cand_tbl)) {
      while (nrow(cand_tbl) > 0) {
        k <- which.min(cand_tbl$score)
        i <- cand_tbl$field_idx[k]; j <- cand_tbl$tls_idx[k]
        inv$f_id[j]          <<- fd$field_id[i]
        inv$f_x[j]           <<- fd$field_x[i]
        inv$f_y[j]           <<- fd$field_y[i]
        inv$f_az[j]          <<- fd$field_az[i]
        inv$f_dist[j]        <<- fd$field_dist[i]
        if (!is.null(fd$field_dbh))    inv$f_dbh[j]    <<- fd$field_dbh[i]
        if (!is.null(fd$field_height)) inv$f_height[j] <<- fd$field_height[i]
        inv$f_score[j]        <<- cand_tbl$score[k]
        inv$f_xy_residual[j]  <<- cand_tbl$xy[k]
        if (!is.null(fd$field_dbh) && !is.na(fd$field_dbh[i]))
          inv$f_delta_dbh[j]    <<- inv$DBH[j]    - fd$field_dbh[i]
        if (!is.null(fd$field_height) && !is.na(fd$field_height[i]))
          inv$f_delta_height[j] <<- inv$Height[j] - fd$field_height[i]
        # Signed azimuth delta in (-180, 180] for bias estimation.
        d <- ((inv$azi[j] - fd$field_az[i] + 540) %% 360) - 180
        inv$f_delta_az[j]     <<- d
        inv$f_delta_dist[j]   <<- inv$distance_m[j] - fd$field_dist[i]
        matched_field[i] <- TRUE
        cand_tbl <- cand_tbl[cand_tbl$field_idx != i & cand_tbl$tls_idx != j, ]
      }
    }
    matched_field
  }

  # Pass 1: wide buffer so rotated trees still produce candidates.
  buf1 <- if (!is.null(tls_params$field_match_buffer_pass1) &&
              isTRUE(tls_params$field_az_bias_correct))
            tls_params$field_match_buffer_pass1
          else tls_params$field_match_buffer
  matched_field <- do_match(buf1)

  # Optional auto-rotation: if first-pass matches reveal a consistent
  # azimuth offset (TLS scanner rotated relative to field compass), apply
  # it and re-match at the tighter final buffer. Uses median to resist outliers.
  if (isTRUE(tls_params$field_az_bias_correct)) {
    n_match <- sum(!is.na(inv$f_id))
    min_deg <- if (!is.null(tls_params$field_az_bias_min_deg))
                 tls_params$field_az_bias_min_deg else 1
    if (n_match >= tls_params$field_az_bias_min_n) {
      bias <- median(inv$f_delta_az[!is.na(inv$f_id)], na.rm = TRUE)
      if (is.finite(bias) && abs(bias) >= min_deg &&
          abs(bias) <= tls_params$field_az_bias_max_deg) {
        message(sprintf("Az bias: rotating inv by %+.2f deg and re-matching at %.2f m.",
                        -bias, tls_params$field_match_buffer))
        inv$azi <<- (inv$azi - bias) %% 360
        inv$X   <<- inv$distance_m * sin(inv$azi * pi / 180)
        inv$Y   <<- inv$distance_m * cos(inv$azi * pi / 180)
        attr(inv, "az_bias_applied") <- bias
        matched_field <- do_match(tls_params$field_match_buffer)
      } else if (is.finite(bias)) {
        message(sprintf("Az bias %.2f deg below threshold or above safety cap; not applied.",
                        bias))
        # Even if no rotation, redo at tighter buffer for the final report.
        if (buf1 != tls_params$field_match_buffer)
          matched_field <- do_match(tls_params$field_match_buffer)
      }
    } else {
      message(sprintf("Az bias correction skipped (only %d matches < min %d).",
                      n_match, tls_params$field_az_bias_min_n))
      if (buf1 != tls_params$field_match_buffer)
        matched_field <- do_match(tls_params$field_match_buffer)
    }
  }

  n_match <- sum(!is.na(inv$f_id))
  message(sprintf("Field match: %d / %d field trees matched (%d TLS unmatched, %d field unmatched). Buffer = %.2f m.",
                  n_match, nrow(fd), nrow(inv) - n_match,
                  nrow(fd) - sum(matched_field),
                  tls_params$field_match_buffer))

  if (n_match) {
    delta_cols <- c("f_xy_residual", "f_delta_dbh", "f_delta_height",
                    "f_delta_az", "f_delta_dist")
    summary_df <- data.frame(
      metric = delta_cols,
      mean   = sapply(inv[delta_cols], mean, na.rm = TRUE),
      sd     = sapply(inv[delta_cols], sd,   na.rm = TRUE),
      rmse   = sapply(inv[delta_cols],
                      function(v) sqrt(mean(v^2, na.rm = TRUE)))
    )
    print(summary_df)
  }

  # Report unmatched field trees so user can investigate.
  if (any(!matched_field)) {
    message(sprintf("Unmatched field trees (%d): %s",
                    sum(!matched_field),
                    paste(fd$field_id[!matched_field], collapse = ", ")))
  }

  write.csv(inv, file.path(out_dir, "inventory_field_matched.csv"),
            row.names = FALSE)
} else {
  message("Section 6 skipped (tls_params$field_xlsx_path is NA or missing).")
}


# ============================================================================
# 7. TODO -- QSM (aRchi)
# ============================================================================
# Parked while we hone segmentation + field validation. Wrapped in if(FALSE)
# so it does not run on a full source(). Flip to if(interactive()) when
# ready to resume LWS tuning + cylinder fitting.
#
# Steps when reactivated:
#   7.1 Per-tree subset (carry eigen + auxiliary signals).
#   7.2 Leaf-wood separation (geometric + optional intensity/HAG/ExG).
#   7.3 QC checkpoint (cross-section / rgl) before committing.
#   7.4 aRchi pipeline: skeletonize -> smooth -> add_radius -> persist.

if (FALSE) {
  qsm_tid <- 78

  # 6.1 Per-tree subset (carry eigen + auxiliary signals) -------------------
  qsm_voxel <- tls_params$qsm_voxel
  qsm_las <- las
  data.table::setDT(qsm_las@data)
  lws_extra <- intersect(c("Intensity", "R", "G", "B"), names(qsm_las@data))
  qsm_las@data <- qsm_las@data[TreeID == qsm_tid,
                               c("X", "Y", "Z",
                                 "Linearity", "Planarity", "Sphericity",
                                 lws_extra), with = FALSE]
  stopifnot(nrow(qsm_las@data) > 0)
  n_full <- nrow(qsm_las@data)

  # 6.2 Leaf-wood separation -------------------------------------------------
  # Geometric base (always on): wood = high Linearity AND low Sphericity AND
  # low Planarity. Optional gates layered on top (each toggled by a CONFIG
  # knob; skipped if the knob is NA or the underlying field isn't present):
  #   Intensity >= lws_intensity_min     (TLS bark vs foliage)
  #   Z         <= lws_hag_max_for_wood  (suppress upper-crown wood FPs)
  #   ExG       <= lws_exg_max           (RGB green-excess; foliage)
  d <- qsm_las@data
  wood_mask <- d$Linearity  >= tls_params$lws_linearity_min &
               d$Sphericity <= tls_params$lws_sphericity_max &
               d$Planarity  <= tls_params$lws_planarity_max
  if (!is.na(tls_params$lws_intensity_min) && "Intensity" %in% names(d))
    wood_mask <- wood_mask & d$Intensity >= tls_params$lws_intensity_min
  if (!is.na(tls_params$lws_hag_max_for_wood))
    wood_mask <- wood_mask & d$Z <= tls_params$lws_hag_max_for_wood
  if (!is.na(tls_params$lws_exg_max) && all(c("R", "G", "B") %in% names(d))) {
    exg <- (2 * as.numeric(d$G) - as.numeric(d$R) - as.numeric(d$B)) /
           max(c(d$R, d$G, d$B), na.rm = TRUE)
    wood_mask <- wood_mask & exg <= tls_params$lws_exg_max
  }
  message(sprintf("[QSM] Tree %d: leaf-wood filter kept %d / %d points (%.1f%% wood).",
                  qsm_tid, sum(wood_mask), n_full,
                  100 * sum(wood_mask) / n_full))

  # 6.3 QC checkpoint -- inspect rgl window before committing to QSM ---------
  # Brown = kept (wood), green = dropped (foliage). If wood looks speckled
  # or trunk has gaps, retune lws_* in CONFIG and re-run from 6.1.
  qc_dt <- data.table::copy(qsm_las@data)[, wood := wood_mask]
  qc_las <- LAS(qc_dt[, .(X, Y, Z)],
                header = qsm_las@header)
  rgl::open3d()
  rgl::bg3d("white")
  rgl::points3d(
    qc_dt$X - mean(qc_dt$X),
    qc_dt$Y - mean(qc_dt$Y),
    qc_dt$Z,
    color = ifelse(qc_dt$wood, "#8B4513", "#228B22"),
    size = 1.5
  )
  rgl::axes3d(c("x--", "y--", "z--"), col = "black")
  rgl::title3d(main = sprintf("Tree %d -- LWS QC (brown=wood, green=foliage)",
                              qsm_tid),
               col = "black")
  rgl::aspect3d("iso")

  # Hard halt: comment out to proceed to QSM once thresholds look right.
  stop("[QSM] LWS QC checkpoint -- inspect rgl window, then comment out this stop() to continue.")

  # 6.4 aRchi pipeline -------------------------------------------------------
  # Commit wood-only subset, thin to scanner-accuracy voxel, then
  # build -> skeletonize -> smooth -> add cylinder radii.
  qsm_las@data <- qsm_las@data[wood_mask, .(X, Y, Z)]
  qsm_las <- decimate_points(qsm_las, random_per_voxel(qsm_voxel, n = 1L))
  rm(d); invisible(gc())
  message(sprintf("[QSM] Tree %d: thinned to %d points (%.0f%% of wood) at %.0f cm voxels.",
                  qsm_tid, nrow(qsm_las@data),
                  100 * nrow(qsm_las@data) / sum(wood_mask), qsm_voxel * 100))

  arc <- aRchi::build_aRchi()
  arc <- aRchi::add_pointcloud(arc, point_cloud = as.data.frame(qsm_las@data))
  arc <- aRchi::skeletonize_pc(arc, D = 0.03, cl_dist = 0.02, max_d = 0.05)
  arc <- aRchi::smooth_skeleton(arc, niter = 1)
  arc <- aRchi::add_radius(arc, sec_length = 0.5, method = "median")

  qsm <- aRchi::get_QSM(arc)
  print(head(qsm))
  message(sprintf("[QSM] %d cylinders | volume = %.4f m3 | trunk DBH = %.3f m",
                  nrow(qsm),
                  aRchi::Treevolume(arc),
                  2 * qsm$radius_cyl[which.min(abs(qsm$startZ - 1.3))]))

  aRchi::plot(arc, show_point_cloud = FALSE)
  aRchi::write_aRchi(arc, file.path(out_dir, sprintf("tree_%d.aRchi", qsm_tid)))
}


# ============================================================================
# (old Section 7 -- field validation -- moved up to Section 6 above)
# ============================================================================

