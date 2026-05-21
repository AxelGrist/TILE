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

# UAV point cloud for fusion (Section 3.6). Set to NULL to skip.
# Must be in a projected CRS (UTM); used as the alignment reference.
uav_input <- "C:/Users/AGRIST/OneDrive - Government of BC/Sklar, Daniel FOR_EX's files - Great Beaver Lake/Plot 311/GBL_ALS.las"   # e.g. "G:/alpine_treemapping/TILE/inputs/uav_gbl.las"
uav_stem_match_radius <- 1.5   # m: UAV stemcls uncertainty + registration residual

# UTM coordinates of plot centre (= SLAM scanner start position).
# SLAM cloud has no CRS; these are added as a coarse pre-translation before ICP.
# Fill in from GPS field notes or first point of the GNSS trajectory.
plot_utm_easting  <- NA_real_   # TODO: UTM easting  (m)
plot_utm_northing <- NA_real_   # TODO: UTM northing (m)
plot_utm_crs      <- NA_integer_  # TODO: EPSG code, e.g. 32610 for UTM Zone 10N


# 1.0 READ----
las_raw <- readLAS(tls_input, select = "xyzicrnRGB") # Only read select headers to save RAM processing
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
# Register treefiltering output column as LAS extra byte so writeLAS() persists it.
las_filtered <- lidR::add_lasattribute(las_filtered, las_filtered@data[["TreeFilterLabel"]], "TreeFilterLabel", "Tree filter label")
writeLAS(las_filtered, file.path(out_dir, "checkpoint_02a_treefilter.laz"))


# 3.2 TreeisoNet: StemCls -> TreeLoc -> shortestpath3D -> CrownOff3D -> CrownClustersSP.
#     Adds TreeID, StemCls, TreeLocX, TreeLocY. Runs on overstory points only.
las_segmented <- treeAIBoxR::treeisonet(
  las_filtered,
  stemcls_model        = "treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)",
  treeloc_model        = "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)",
  crownoff_model       = "treeisonet_tls_boreal_crownoff_esegformer3D_128_15cm(GPU4GB)",
  conf_thresh          = 0.15,  # Python hardcodes 0.1; raised to 0.15 to suppress sapling phantoms (trees 109,162,198,221,406,526)
  max_isolated_dist    = 0.12,  # default 0.3 m - 3x voxel size; bridges stem/crown gaps without jumping trees
  k_graph              = 10L,   # Python stemCluster.shortestpath3D uses k=min(len,10) hardcoded
  k_node               = 10L,   # Python stemCluster.create_node_graph() uses k=10 hardcoded; R default was 20
  stemcls_min_points   = 200L,  # Python default 100; raised to 200 as second gate against sapling stem clusters
  cuda                 = TRUE,
  verbose              = TRUE
)

# 3.3 Trim: drop trees whose TreeLoc base falls outside plot_radius + plot_buffer.
#     After trim, las contains only points assigned to plot trees.
r_max     <- plot_radius + plot_buffer
tree_locs <- las_segmented@data[TreeID > 0L & !is.na(TreeLocX),
                                .(lx = TreeLocX[1L], ly = TreeLocY[1L]),
                                by = TreeID]
plot_ids  <- tree_locs[sqrt((lx - plot_center_x)^2 + (ly - plot_center_y)^2) <= r_max, TreeID]
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

# Register treeisonet output columns as LAS extra bytes so writeLAS() persists them.
las_segmented <- lidR::add_lasattribute(las_segmented, las_segmented@data[["TreeID"]],   "TreeID",   "Tree ID")
las_segmented <- lidR::add_lasattribute(las_segmented, las_segmented@data[["StemCls"]],  "StemCls",  "Stem class")
las_segmented <- lidR::add_lasattribute(las_segmented, las_segmented@data[["TreeLocX"]], "TreeLocX", "Tree base X (m)")
las_segmented <- lidR::add_lasattribute(las_segmented, las_segmented@data[["TreeLocY"]], "TreeLocY", "Tree base Y (m)")

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

# Run these lines (+ config) if needed to recover las checkpoint after a crash
#las_segmented <- lidR::readLAS(file.path(out_dir, "checkpoint_02b_segmented.laz"))
#data.table::setDT(las_segmented@data)


# ============================================================================
# 3.6  UAV FUSION
# Register TLS into UAV CRS, run UAV treeisonet, match stems, fuse clouds.
# If no UAV data: run the TLS-only fallback line below and skip to Section 4.
# ============================================================================

# TLS-only fallback — run this and skip to Section 4 if uav_input is not used.
las_fused <- las_segmented
data.table::setDT(las_fused@data)
las_fused@data[, Source := 1L]

# 3.6a  Register TLS into UAV CRS  (UAV = reference, TLS = moving)
utm_crs <- sf::st_crs(lidR::readLASheader(uav_input))

# Pre-translate SLAM cloud from scanner-local frame to UTM.
# SLAM origin (0,0) = scanner start = plot centre, so offset = plot GPS coordinates.
las_pretrans <- lidR::readLAS(file.path(out_dir, "checkpoint_02b_segmented.laz"))
data.table::setDT(las_pretrans@data)
las_pretrans@data[, X := X + plot_utm_easting]
las_pretrans@data[, Y := Y + plot_utm_northing]
lidR::projection(las_pretrans) <- sf::st_crs(plot_utm_crs)
lidR::writeLAS(las_pretrans, file.path(out_dir, "tls_pretranslated.laz"))
rm(las_pretrans)
# QC: lidR::plot(lidR::readLAS(file.path(out_dir, "tls_pretranslated.laz")), color = "Z", bg = "white")

al <- lidRalignment::AlignmentScene$new(
        uav_input,
        file.path(out_dir, "tls_pretranslated.laz"))
al$set_ref_is_ground_based(FALSE)   # UAV = airborne
al$set_mov_is_ground_based(TRUE)    # TLS = ground-based
al$set_radius(plot_radius + plot_buffer)
al$align()

M            <- al$get_registration_matrix()
tmp_reg      <- lidRalignment::transform_las(
                  file.path(out_dir, "checkpoint_02b_segmented.laz"), M, utm_crs)
tls_reg_path <- file.path(out_dir, "checkpoint_02c_tls_registered.laz")
file.rename(tmp_reg, tls_reg_path)

# Patch extra bytes: transform_las strips all non-standard fields.
las_reg_stripped <- lidR::readLAS(tls_reg_path)
las_seg_orig     <- lidR::readLAS(file.path(out_dir, "checkpoint_02b_segmented.laz"))
las_reg_stripped <- lidR::add_lasattribute(las_reg_stripped, las_seg_orig@data[["TreeID"]],   "TreeID",   "Tree ID")
las_reg_stripped <- lidR::add_lasattribute(las_reg_stripped, las_seg_orig@data[["StemCls"]],  "StemCls",  "Stem class")
las_reg_stripped <- lidR::add_lasattribute(las_reg_stripped, las_seg_orig@data[["TreeLocX"]], "TreeLocX", "Tree base X (m)")
las_reg_stripped <- lidR::add_lasattribute(las_reg_stripped, las_seg_orig@data[["TreeLocY"]], "TreeLocY", "Tree base Y (m)")
lidR::writeLAS(las_reg_stripped, tls_reg_path)
rm(las_reg_stripped, las_seg_orig)

las_tls_reg <- lidR::readLAS(tls_reg_path)
data.table::setDT(las_tls_reg@data)
las_tls_reg@data[, Source := 1L]
# QC: lidR::plot(las_tls_reg, color = "Z", bg = "white")

tls_ctr_x  <- (las_tls_reg@header$`Min X` + las_tls_reg@header$`Max X`) / 2
tls_ctr_y  <- (las_tls_reg@header$`Min Y` + las_tls_reg@header$`Max Y`) / 2
uav_clip_r <- plot_radius + plot_buffer + 5   # 5 m margin beyond plot edge

# 3.6b  UAV treeisonet  (clipped to plot extent before segmentation)
las_uav_raw <- lidR::readLAS(uav_input,
                             filter = paste("-keep_circle", tls_ctr_x, tls_ctr_y, uav_clip_r))
message(sprintf("[3.6b] UAV clip: %d pts.", npoints(las_uav_raw)))
# QC: lidR::plot(las_uav_raw, color = "Z", bg = "white")

las_uav_seg <- treeAIBoxR::treeisonet(
  las_uav_raw,
  sensor         = "uav",
  stemcls_model  = "treeisonet_uav_mixedwood_stemcls_esegformer3D_128_8cm(GPU3GB)",
  treeloc_model  = "treeisonet_uav_mixedwood_treeloc_esegformer3D_128_10cm(GPU3GB)",
  crownoff_model = "treeisonet_uav_mixedwood_crownoff_esegformer3D_128_15cm(GPU4GB)",
  cuda           = TRUE,
  verbose        = TRUE
)
las_uav_seg <- lidR::add_lasattribute(las_uav_seg, las_uav_seg@data[["TreeID"]],   "TreeID",   "Tree ID")
las_uav_seg <- lidR::add_lasattribute(las_uav_seg, las_uav_seg@data[["StemCls"]],  "StemCls",  "Stem class")
las_uav_seg <- lidR::add_lasattribute(las_uav_seg, las_uav_seg@data[["TreeLocX"]], "TreeLocX", "Tree base X (m)")
las_uav_seg <- lidR::add_lasattribute(las_uav_seg, las_uav_seg@data[["TreeLocY"]], "TreeLocY", "Tree base Y (m)")
lidR::writeLAS(las_uav_seg, file.path(out_dir, "checkpoint_uav_segmented.laz"))
data.table::setDT(las_uav_seg@data)
las_uav_seg@data[, Source := 2L]
# QC: lidR::plot(las_uav_seg, color = "TreeID", colorPalette = rainbow(50), bg = "white")

# 3.6c  Stem matching  (LSAP — globally optimal 1:1 assignment)
# TLS stems: XY from lower 25% of cloud (more stable than TreeLocX/Y in scanner CRS post-registration)
tls_stems <- las_tls_reg@data[!is.na(TreeID) & TreeID > 0L,
  {
    z_lo <- min(Z) + (max(Z) - min(Z)) * 0.25   # lower quartile of tree height
    .(X_stem = mean(X[Z <= z_lo]), Y_stem = mean(Y[Z <= z_lo]), H_m = max(Z) - min(Z))
  },
  by = TreeID]

uav_stems <- las_uav_seg@data[!is.na(TreeID) & TreeID > 0L & !is.na(TreeLocX),
               .(X_stem = mean(TreeLocX), Y_stem = mean(TreeLocY), H_m = max(Z) - min(Z)),
               by = TreeID]

tls_std <- TreeMatching::standardize(as.data.frame(tls_stems), "X_stem", "Y_stem", "H_m", "m", crs = utm_crs, idname = "TreeID")
uav_std <- TreeMatching::standardize(as.data.frame(uav_stems), "X_stem", "Y_stem", "H_m", "m", crs = utm_crs, idname = "TreeID")

plot_ctr_x <- mean(range(tls_stems$X_stem))
plot_ctr_y <- mean(range(tls_stems$Y_stem))
plot_r_m   <- max(sqrt((tls_stems$X_stem - plot_ctr_x)^2 + (tls_stems$Y_stem - plot_ctr_y)^2)) + 5

treemap <- TreeMatching::make_mapmatching(tls_std, uav_std, center = c(plot_ctr_x, plot_ctr_y), radius = plot_r_m)
treemap <- TreeMatching::match_trees(treemap, dxymax = uav_stem_match_radius, dzmax = 50, zrel = 40)
# QC: plot(treemap, scale = 2)
print(treemap)

mt      <- data.table::as.data.table(treemap$match_table)
matches <- mt[state == "Matched", .(tls_TreeID = id_inventory, uav_TreeID = id_measure)]
message(sprintf("[3.6c] Matched %d / %d TLS trees (%.0f%%) [LSAP].",
                nrow(matches), nrow(tls_stems), 100 * nrow(matches) / nrow(tls_stems)))

# 3.6d  Fuse: relabel matched UAV points to TLS TreeID; new IDs for unmatched UAV trees
uav_data <- data.table::copy(las_uav_seg@data)
uav_data[TreeID %in% matches$uav_TreeID, TreeID := matches$tls_TreeID[match(TreeID, matches$uav_TreeID)]]

unmatched_uav_ids <- setdiff(
  las_uav_seg@data[TreeID > 0L & !is.na(TreeID), unique(TreeID)],
  matches$uav_TreeID)
if (length(unmatched_uav_ids) > 0L) {
  max_id  <- max(las_tls_reg@data$TreeID, na.rm = TRUE)
  new_map <- setNames(seq_along(unmatched_uav_ids) + max_id, as.character(unmatched_uav_ids))
  uav_data[TreeID %in% unmatched_uav_ids, TreeID := new_map[as.character(TreeID)]]
  message(sprintf("[3.6d] %d unmatched UAV trees \u2192 new IDs %d\u2013%d.",
                  length(unmatched_uav_ids), max_id + 1L, max_id + length(unmatched_uav_ids)))
}
las_uav_seg@data <- uav_data

las_fused <- rbind(las_tls_reg, las_uav_seg)
las_fused <- lidR::add_lasattribute(las_fused, las_fused@data[["Source"]], "Source", "Point source (1=TLS 2=UAV)")
lidR::writeLAS(las_fused, file.path(out_dir, "checkpoint_02d_fused.laz"))
message(sprintf("[3.6d] Fused: %d TLS + %d UAV = %d pts, %d trees.",
                npoints(las_tls_reg), npoints(las_uav_seg), npoints(las_fused),
                las_fused@data[TreeID > 0L & !is.na(TreeID), data.table::uniqueN(TreeID)]))
# QC: lidR::plot(las_fused, color = "Source", colorPalette = c("steelblue", "tomato"), bg = "white")







# ============================================================================
# 4. WoodCls  (wood / foliage classification)
# Run per source: TLS 2.5 cm model on TLS points, UAV 8 cm model on UAV points.
# Missing columns (e.g. StemCls absent from UAV) are filled with NA on rbind.
# ============================================================================
las_tls_pts <- lidR::filter_poi(las_fused, Source == 1L)
las_uav_pts <- lidR::filter_poi(las_fused, Source == 2L)

# TLS WoodCls
las_tls_wc <- treeAIBoxR::woodcls(
  las_tls_pts,
  model     = "woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)",
  sensor    = "tls",
  component = "branch",
  cuda      = TRUE,
  verbose   = TRUE
)
las_tls_wc <- treeAIBoxR::woodcls(
  las_tls_wc,
  model               = "woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)",
  component           = "downed_log",
  downed_z_max        = 1.5,    # m above ground (Xi 2023)
  downed_tilt_min_deg = 60.0,   # deg from vertical (Xi 2023)
  cuda                = TRUE,
  verbose             = TRUE
)

# UAV WoodCls (8 cm model; skipped if no UAV points present)
if (npoints(las_uav_pts) > 0L) {
  las_uav_wc <- treeAIBoxR::woodcls(
    las_uav_pts,
    model     = "woodcls_branch_uav_esegformer3D_128_8cm(GPU3GB)",
    sensor    = "uav",
    component = "branch",
    cuda      = TRUE,
    verbose   = TRUE
  )
} else {
  las_uav_wc <- las_uav_pts
}

# Merge and assign 8-class forest components.
# UAV points: StemCls = NA -> ComponentClass derived from WoodLabel alone.
las_woodcls <- rbind(las_tls_wc, las_uav_wc)
las_woodcls <- treeAIBoxR::forest_components(las_woodcls)

# Register woodcls/forest_components output columns as LAS extra bytes so writeLAS() persists them.
las_woodcls <- lidR::add_lasattribute(las_woodcls, las_woodcls@data[["WoodLabel"]],      "WoodLabel",      "Wood label")
las_woodcls <- lidR::add_lasattribute(las_woodcls, las_woodcls@data[["DownedLog"]],      "DownedLog",      "Downed log flag")
las_woodcls <- lidR::add_lasattribute(las_woodcls, las_woodcls@data[["ComponentClass"]], "ComponentClass", "Forest component class")
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
