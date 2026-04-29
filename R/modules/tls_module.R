# TLS MODULE: Terrestrial Laser Scanner Processing ----
#
# 1. Inspect cloud (lidR readLAS / las_check).
# 2. Pre-process at full density: dedupe, ground (CSF), normalize.
# 3. Tree segmentation via Comparative Shortest Path (Tao 2015 / Larysch 2025).
# 4. Forest inventory: DBH, height, position, crown projection per tree.
# 5. QSM (aRchi) + structural metrics (ITSMe) -- to follow.
#
# Refs:
#   https://r-lidar.github.io/lidRbook/gnd.html
#   https://cran.r-project.org/package=CspStandSegmentation
#   Tao et al. 2015 ISPRS J. Photogramm. Remote Sens. 110:66-76
#   Larysch et al. 2025 doi:10.1007/s10342-025-01796-z

library(lidR)
library(CspStandSegmentation)
library(aRchi)
library(ITSMe)
library(sf)

# Parallelism -- Xeon w5-2545: 12 physical / 24 logical cores, 128 GB RAM.
# Use ALL logical threads; this box has nothing else competing for cycles.
n_threads <- parallel::detectCores(logical = TRUE)   # 24

# lidR (OpenMP): classify_ground, normalize_height, filter_*, decimate_points,
# add_attribute, plot, etc.
lidR::set_lidr_threads(n_threads)

# data.table is used internally by lidR for LAS payloads; default is 50% of
# cores -- lift it to match.
if (requireNamespace("data.table", quietly = TRUE))
  data.table::setDTthreads(n_threads)

# BLAS/LAPACK thread cap is set in TILE/.Renviron (OMP/OPENBLAS/MKL_NUM_THREADS=24)
# and applied at R startup -- restart R if you change it.
options(lidR.progress = TRUE)


# CONFIG ----

tls_input <- "C:/Users/AGRIST/OneDrive - Government of BC/Sklar, Daniel FOR_EX's files - Great Beaver Lake/Plot 311/20250611101844578.las"
out_dir   <- "G:/alpine_treemapping/TILE/outputs/tls/Plot_311"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Ground algorithm: CSF chosen for this plot (dense low understory).
# CSF treats the cloud as having a draped cloth from below; with a rigid
# cloth + tight class_threshold it follows true ground beneath shrubs
# instead of latching onto shrub tops the way PTD seed selection does.
tls_params <- list(
  # Ground (CSF -- chosen for dense low understory).
  csf_class_threshold  = 0.05,
  csf_cloth_resolution = 0.4,
  csf_rigidness        = 3L,
  csf_sloop_smooth     = TRUE,
  csf_time_step        = 0.65,
  # Height filter (after normalization)
  z_min                = 0.2,
  z_max                = 50,

  # Noise removal (Statistical Outlier Removal: drop points whose mean
  # distance to k neighbors is > m * sd above the cloud-wide mean).
  sor_k                = 8,      # neighbors used in mean-distance estimate
  sor_m                = 3,      # std-dev multiplier (lower = more aggressive)

  # Sub-canopy foliage filter (eigen-feature based): below shrub_height_max,
  # drop points whose local geometry is foliage/leaf-like (planar OR spherical)
  # AND not stem-like (low linearity). Stems / branches are linear, foliage
  # is planar (leaves, fronds) or spherical (needle clusters, noise).
  # Requires Geometry features from add_geometry().
  shrub_height_max     = 4.0,    # m; only filter below this height
  shrub_planarity_min  = 0.4,    # >= this is leaf-like
  shrub_sphericity_min = 0.35,   # >= this is needle-cluster-like
  shrub_linearity_max  = 0.4,    # <= this is non-stem-like (protects stems)

  # Stem base detection (find_base_coordinates_geom -- shape-aware:
  # filters slice points to vertical+planar before 3D-clustering. Better
  # than the raster variant when stems are close (3D clustering won't
  # bridge two trunks the way 2D density cells can) and on plots with
  # mixed stem sizes (no global density quantile to drown out small stems).
  # Trade-off: slower (per-point eigen) and 4 knobs instead of 2.
  base_zmin            = 1.0,    # m; above shrub layer
  base_zmax            = 2.0,    # m; below most branching
  base_res             = 0.06,   # m; 3D DBSCAN eps -- tight enough to keep
                                 #    close thin stems (~7 cm DBH, 15 cm gap)
                                 #    in separate clusters
  base_min_vert        = 0.85,   # >= verticality to keep as trunk-like
  base_min_plan        = 0.50,   # >= planarity  to keep as trunk-like
  base_min_cluster     = 50,     # min points per cluster to count as a stem
  # Raster fallback (uncomment + swap call in Section 3 if geom misbehaves):
  # base_res             = 0.05,   # m; fine raster -- separate close stems
  # base_density_q       = 0.99,   # top 1% density cells flagged
  # base_merge_eps       = 0.15,   # m; collapse duplicates within 15 cm
  # Tree filter (post-inventory: drop shrubs/saplings/branch fragments)
  tree_min_dbh         = 0.05,   # m; >=5 cm DBH to count as a tree
  tree_min_height      = 5.0,    # m; >=5 m tall to count as a tree
  # Plot extent filter (drop trees outside the buffered plot radius)
  plot_center_x        = 0.0,    # m; plot center X (scanner-local frame)
  plot_center_y        = 0.0,    # m; plot center Y
  plot_radius          = 11.28,  # m; standard plot radius
  plot_buffer          = 1.72,   # m; extra buffer (-> 13 m total)
  # CSP segmentation -- accuracy-tuned
  csp_voxel            = 0.07,   # m; fine routing graph -- closer to scanner
                                 #    accuracy floor; reduces sideways path
                                 #    shortcuts that flip TreeID on leaners
  csp_v_w              = 0.5,    # verticality weight; moderate -- too high
                                 #    penalizes leaning trunks and routes them
                                 #    into neighbors' vertical stems
  csp_l_w              = 0.3,    # linearity weight; mild penalty for branches
  csp_s_w              = 0.7,    # sphericity weight; strongly penalize foliage
  csp_n_cores          = parallel::detectCores(logical = TRUE),
  # TLS scanner accuracy (upper bound of inter-frame error). For GBL Plot 311
  # the SLAM report shows 100% of points within 5 cm of true surfaces, so any
  # spacing finer than this is geometrically redundant. Drives `inv_thin_voxel`
  # and `qsm_voxel` so they always sit at or below the noise floor.
  tls_accuracy_m       = 0.05,   # m; SLAM/TLS upper error bound
  # Forest inventory (taper curve fit, full forest_inventory)
  inv_slice_min        = 0.3,    # m; bottom of taper band
  inv_slice_max        = 4.0,    # m; top of taper band
  inv_increment        = 0.1,    # m; fine vertical step (more taper detail)
  inv_width            = 0.05,   # m; thin slices for clean circle fits
  inv_max_dbh          = 1.0,    # m; reject impossibly large stems
  inv_n_cores          = parallel::detectCores(logical = TRUE)
)

# Derived voxel sizes -- never exceed the scanner accuracy.
# Inventory: 1/5th of accuracy keeps fine taper detail (well below noise) and
#            keeps per-slice point counts under ~40k -- forest_inventory() has
#            a quantile-filter bug (q = 1 - sqrt(100/N) + 0.05) that fails
#            when N >~40k. Going finer (e.g. /10) re-triggers the bug.
# QSM:       2/5th of accuracy is the densest spacing that still adds info.
tls_params$inv_thin_voxel <- min(0.01, tls_params$tls_accuracy_m / 5)
tls_params$qsm_voxel      <- min(0.02,  tls_params$tls_accuracy_m * 0.4)


# 1. READ + INSPECT ----

las <- readLAS(tls_input, select = "xyzicrnRGB")  # keep core attrs + RGB
if (is.empty(las)) stop("Input LAS is empty / failed to read.")

las_check(las)
print(las)

las <- filter_duplicates(las)


# 2. PRE-PROCESS (full density) ----

las <- classify_ground(las, csf(
  sloop_smooth     = tls_params$csf_sloop_smooth,
  class_threshold  = tls_params$csf_class_threshold,
  cloth_resolution = tls_params$csf_cloth_resolution,
  rigidness        = tls_params$csf_rigidness,
  time_step        = tls_params$csf_time_step
))

las <- normalize_height(las, tin())
las <- filter_poi(las, Z >= tls_params$z_min, Z <= tls_params$z_max)

# 2a. Statistical outlier removal -- drop scanner ghost / mixed-pixel noise.
# add_geometry() and density-raster base detection are both sensitive to
# isolated outliers; SOR removes them cheaply.
npts0 <- npoints(las)
las <- classify_noise(las, sor(k = tls_params$sor_k, m = tls_params$sor_m))
las <- filter_poi(las, Classification != LASNOISE)
message(sprintf("SOR: removed %d / %d points (%.2f%%) as noise.",
                npts0 - npoints(las), npts0,
                100 * (npts0 - npoints(las)) / npts0))


# 3. STEM BASES + CSP SEGMENTATION ----
# Tao 2015 CSP: builds a connectivity graph over voxelized cloud, computes
# weighted shortest paths from every voxel to every detected stem base, and
# assigns each point to its graph-nearest stem. Path costs are weighted by
# eigen geometry (verticality, sphericity) so routing prefers stem-like
# structures and avoids leaking through foliage into neighbor crowns.

# Eigen geometry features (verticality, planarity, linearity, sphericity)
# required by both the base finder and the CSP cost function.
las <- add_geometry(las, n_cores = tls_params$csp_n_cores)

# Sub-canopy foliage filter: below shrub_height_max, drop points whose
# local geometry is leaf-like (Planarity high) OR needle-cluster-like
# (Sphericity high), AND not stem-like (Linearity low). This removes
# shrubs / fern fronds / low foliage / coarse woody debris without
# touching tree stems (which are linear, not planar or spherical).
npts0 <- npoints(las)
shrub_layer <- las@data$Z < tls_params$shrub_height_max
foliage_shape <- (las@data$Planarity  >= tls_params$shrub_planarity_min |
                  las@data$Sphericity >= tls_params$shrub_sphericity_min) &
                  las@data$Linearity  <= tls_params$shrub_linearity_max
las <- filter_poi(las, !(shrub_layer & foliage_shape))
message(sprintf("Eigen shrub filter: removed %d / %d points (%.2f%%) below %.1f m.",
                npts0 - npoints(las), npts0,
                100 * (npts0 - npoints(las)) / npts0,
                tls_params$shrub_height_max))

# Detect stem bases via shape-aware geometry: keep slice points that are
# vertical AND planar (trunk-like), then 3D DBSCAN-cluster them. 3D
# clustering separates leaning/close stems that 2D density would merge.
#
# Local fast variant: CspStandSegmentation::find_base_coordinates_geom()
# unconditionally calls add_geometry() on the slice, which is wasteful here
# because we already computed eigen features on the full cloud above.
# This function reuses those columns if present, otherwise falls back to
# the same add_geometry() call as upstream. Logic is otherwise identical
# to find_base_coordinates_geom (CspStandSegmentation 0.2.0).
find_bases_geom_fast <- function(las, zmin, zmax, res,
                                 min_verticality, min_planarity,
                                 min_cluster_size) {
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
  data.frame(X = xy[, 2], Y = xy[, 3], Z = z[, 2],
             TreeID = seq_len(nrow(xy)))
}

bases <- find_bases_geom_fast(las,
  zmin             = tls_params$base_zmin,
  zmax             = tls_params$base_zmax,
  res              = tls_params$base_res,
  min_verticality  = tls_params$base_min_vert,
  min_planarity    = tls_params$base_min_plan,
  min_cluster_size = tls_params$base_min_cluster
)
# Upstream (recomputes eigen on slice -- slower but identical result):
# bases <- find_base_coordinates_geom(las, zmin=..., zmax=..., res=...,
#   min_verticality=..., min_planarity=..., min_cluster_size=...)
# Raster fallback (density-based, faster, but merges close stems):
# bases <- find_base_coordinates_raster(las,
#   res  = tls_params$base_res, zmin = tls_params$base_zmin,
#   zmax = tls_params$base_zmax, q = tls_params$base_density_q,
#   eps  = tls_params$base_merge_eps)

# Drop bases outside the buffered plot radius BEFORE CSP -- avoids routing
# the entire cloud to seeds we'll discard later, saving Dijkstra time
# proportional to the number of removed bases.
nb0 <- nrow(bases)
base_dist <- sqrt((bases$X - tls_params$plot_center_x)^2 +
                  (bases$Y - tls_params$plot_center_y)^2)
bases <- bases[base_dist <= (tls_params$plot_radius + tls_params$plot_buffer), ]
message(sprintf("Plot-extent base filter: kept %d / %d bases inside %.2f m + %.2f m buffer.",
                nrow(bases), nb0, tls_params$plot_radius, tls_params$plot_buffer))

# CSP segmentation -- assigns TreeID to every point.
las <- csp_cost_segmentation(las, bases,
  Voxel_size = tls_params$csp_voxel,
  V_w        = tls_params$csp_v_w,
  L_w        = tls_params$csp_l_w,
  S_w        = tls_params$csp_s_w,
  N_cores    = tls_params$csp_n_cores
)


# 4. FOREST INVENTORY ----
# Full taper-curve inventory: fits a circle in 10 cm slices every 20 cm
# from 0.3 to 4 m, then splines DBH/X/Y vs Z. Returns one row per TreeID
# with X, Y, DBH, Height, ConvexHullArea, quality_flag.
#
# forest_inventory() has a quantile-filter bug that fails when individual
# slices contain >~40k points (q > 1 -> quantile() error). TLS density
# easily exceeds this. Workaround: thin a COPY of the cloud (voxel size
# derived from tls_params$tls_accuracy_m, see CONFIG) for inventory only --
# preserves circle-fit fidelity (voxel << scanner noise) while keeping
# `las` full-density for downstream QSM/ITSMe.

inv_input <- decimate_points(las, random_per_voxel(tls_params$inv_thin_voxel, n = 1L))

inv <- forest_inventory(inv_input,
  slice_min = tls_params$inv_slice_min,
  slice_max = tls_params$inv_slice_max,
  increment = tls_params$inv_increment,
  width     = tls_params$inv_width,
  max_dbh   = tls_params$inv_max_dbh,
  n_cores   = tls_params$inv_n_cores
)

rm(inv_input); gc()

# Drop shrubs / saplings / branch fragments: keep only valid trees.
# (Plot-extent filter was already applied to bases pre-CSP.)
inv <- inv[!is.na(inv$DBH) &
           inv$DBH    >= tls_params$tree_min_dbh &
           inv$Height >= tls_params$tree_min_height, ]

message(sprintf("Inventory: %d trees retained after DBH/Height filter.", nrow(inv)))

write.csv(inv, file.path(out_dir, "inventory.csv"), row.names = FALSE)
writeLAS(las,  file.path(out_dir, "plot_classified.laz"))


# 5. INTERACTIVE PLOTTING ----
# Whole-plot view: every TreeID, with DBH/Height labels overlaid.
# Single-tree view: filter to one TreeID, plot at full density.

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
  las_seg <- color_ids(las_seg)

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
}




# 6. QSM (aRchi) -----------------------------------------------------------
# Per-tree QSM: subset full-density `las` by TreeID, build aRchi skeleton,
# fit cylinder radii, then visualize / extract metrics.

if (interactive()) {
  qsm_tid <- 77

  # 1. Subset to one tree from the full-density `las`, then thin to a voxel
  #    derived from `tls_params$tls_accuracy_m` (see CONFIG above). Spacing
  #    finer than the scanner noise floor is redundant for cylinder fitting.
  qsm_voxel <- tls_params$qsm_voxel
  qsm_las <- las
  data.table::setDT(qsm_las@data)
  qsm_las@data <- qsm_las@data[TreeID == qsm_tid, .(X, Y, Z)]
  stopifnot(nrow(qsm_las@data) > 0)
  n_full <- nrow(qsm_las@data)
  qsm_las <- decimate_points(qsm_las, random_per_voxel(qsm_voxel, n = 1L))
  message(sprintf("[QSM] Tree %d: %d -> %d points (%.0f%%) at %.0f cm voxels",
                  qsm_tid, n_full, nrow(qsm_las@data),
                  100 * nrow(qsm_las@data) / n_full, qsm_voxel * 100))

  # 2. aRchi pipeline. Build -> skeletonize (graph-based stem/branch network)
  #    -> smooth -> add cylinder radii by axis section.
  arc <- aRchi::build_aRchi()
  arc <- aRchi::add_pointcloud(arc, point_cloud = as.data.frame(qsm_las@data))
  arc <- aRchi::skeletonize_pc(arc, D = 0.03, cl_dist = 0.02, max_d = 0.05)
  arc <- aRchi::smooth_skeleton(arc, niter = 1)
  arc <- aRchi::add_radius(arc, sec_length = 0.5, method = "median")

  # 3. Inspect.
  qsm <- aRchi::get_QSM(arc)
  print(head(qsm))
  message(sprintf("[QSM] %d cylinders | volume = %.4f m3 | trunk DBH = %.3f m",
                  nrow(qsm),
                  aRchi::Treevolume(arc),
                  2 * qsm$radius_cyl[which.min(abs(qsm$startZ - 1.3))]))

  # 4. 3D view: skeleton + cylinders.
  aRchi::plot(arc, show_point_cloud = FALSE)

  # 5. Persist for later (ITSMe etc.).
  aRchi::write_aRchi(arc, file.path(out_dir, sprintf("tree_%d.aRchi", qsm_tid)))
}
