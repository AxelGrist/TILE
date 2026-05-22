# ============================================================================
# TLS MODULE: Terrestrial Laser Scanner Processing
# ============================================================================
# Pipeline:
#   1. Read + inspect cloud (lidR readLAS / las_check).
#   2. Pre-process: dedupe, SOR, then decimate to scanner precision
#      (res = 0.02 m, RS10). No ground classification, height normalization,
#      or height filter needed: all DL models voxelize per-block relative to
#      local block minimum, so they are invariant to absolute Z values.
#   2.5 Registration (optional): align TLS to UAV CRS via ICP (lidRalignment).
#       Skip if no UAV — downstream runs in scanner-local frame.
#   3. Segmentation: TreeFilter -> overstory mask; TreeisoNet (StemCls ->
#      TreeLoc -> shortestpath3D) -> TreeID per overstory point.
#   3.6 UAV fusion (optional): UAV treeisonet + LSAP stem matching + merge.
#       Skip if no UAV — TLS-only fallback assigns Source = 1.
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
plot_ctr_x    <- plot_center_x   # active plot centre — overridden to UTM by Section 2.5 if UAV available
plot_ctr_y    <- plot_center_y
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
# Used to pre-clip the UAV tile to the plot area before ICP alignment.
# Derived from first GNSS trajectory point (lon=-123.47162668, lat=54.41643039).
plot_utm_easting  <- 469394.564   # UTM Zone 10N easting  (m)
plot_utm_northing <- 6029958.118  # UTM Zone 10N northing (m)


# 1.0 READ----
las_raw <- readLAS(tls_input, select = "xyzicrnRGB") # Only read select headers to save RAM processing
las_check(las_raw)

# 2.0 PRE-PROCESSING----

# 2.1 Point Deduplication
las_preproc <- filter_duplicates(las_raw)

# 2.2 Statistical outlier removal
npts0 <- npoints(las_preproc)
las_preproc <- classify_noise(las_preproc, sor(k = 10, m = 1))
las_preproc <- filter_poi(las_preproc, Classification != LASNOISE)
message(sprintf("SOR: removed %d / %d points (%.2f%%) as noise.",
                npts0 - npoints(las_preproc), npts0,
                100 * (npts0 - npoints(las_preproc)) / npts0))

# 2.3 Point decimation (3D) — one random point per 2 cm voxel.
# RS10 SLAM inter-frame error ~2 cm; no sub-2cm geometry is recoverable.
# Decimation before ICP produces more uniform point spacing, which improves
# correspondence quality vs. the raw density-gradient cloud.
npts1 <- npoints(las_preproc)
las_preproc <- decimate_points(las_preproc, random_per_voxel(res = 0.02))
message(sprintf("Decimate: %d -> %d points (%.1f%% retained).",
                npts1, npoints(las_preproc), 100 * npoints(las_preproc) / npts1))

writeLAS(las_preproc, file.path(out_dir, "checkpoint_01_preproc.laz"))
# Recovery: las_preproc <- lidR::readLAS(file.path(out_dir, "checkpoint_01_preproc.laz"))

# ============================================================================
# 2.5  TLS REGISTRATION  (skip to Section 3 if no UAV data)
# Aligns TLS to UAV CRS via ICP; all downstream sections then operate in UTM
# so TreeLocX/Y from treeisonet are georeferenced natively.
# TLS-only path: skip this section — plot_ctr_x/y remain at scanner-local (0, 0).
# ============================================================================
utm_crs <- sf::st_crs(lidR::readLASheader(uav_input))

# Pre-clip UAV tile to plot area.
# GBL_ALS.las is a ~632m x 389m tile; its centroid is ~211m from the plot.
# lidRalignment clips each cloud around its own centroid, so feeding the full
# tile would clip to the wrong location. Pre-clipping ensures both clouds
# cover the same real-world area before the centroid normalisation step.
uav_plot_clip_path <- file.path(out_dir, "uav_plot_clip.laz")
las_uav_clip <- lidR::readLAS(uav_input,
                 filter = paste("-keep_circle",
                                plot_utm_easting, plot_utm_northing,
                                plot_radius + plot_buffer + 10))
lidR::writeLAS(las_uav_clip, uav_plot_clip_path)
rm(las_uav_clip)

# Align TLS (all points: ground + stems + canopy) to UAV clip.
# las_preproc is reloaded after registration so Section 3 treefiltering
# and treeisonet operate in UTM without further code changes.
alignment <- lidRalignment::AlignmentScene$new(
              uav_plot_clip_path,
              file.path(out_dir, "checkpoint_01_preproc.laz"))
alignment$set_ref_is_ground_based(FALSE)   # UAV = airborne
alignment$set_mov_is_ground_based(TRUE)    # TLS = ground-based
alignment$set_radius(plot_radius + plot_buffer)
alignment$align()
# QC: alignment$plot("raw"); alignment$plot("coarse"); alignment$plot("fine"); alignment$plot("extra", compare_to = "fine")

transform_matrix <- alignment$get_registration_matrix()
tmp_reg          <- lidRalignment::transform_las(
                     file.path(out_dir, "checkpoint_01_preproc.laz"), transform_matrix, utm_crs)
file.rename(tmp_reg, file.path(out_dir, "checkpoint_02_tls_registered.laz"))
# QC: lidR::plot(lidR::readLAS(file.path(out_dir, "checkpoint_02_tls_registered.laz")), color = "Z", bg = "white")

# Reload as las_preproc so downstream sections are CRS-agnostic.
las_preproc <- lidR::readLAS(file.path(out_dir, "checkpoint_02_tls_registered.laz"))
plot_ctr_x  <- plot_utm_easting    # override scanner-local (0, 0) with UTM
plot_ctr_y  <- plot_utm_northing

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
writeLAS(las_filtered, file.path(out_dir, "checkpoint_03a_treefilter.laz"))


# 3.2 TreeisoNet: StemCls -> TreeLoc -> shortestpath3D -> CrownOff3D -> CrownClustersSP.
#     Adds TreeID, StemCls, TreeLocX, TreeLocY. Runs on overstory points only.
#     Out-of-plot trees form their own segments; Section 3.3 deletes them, preserving
#     full crown geometry for in-plot trees near the boundary.
las_segmented <- treeAIBoxR::treeisonet(
  las_filtered,
  stemcls_model        = "treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)",
  treeloc_model        = "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)",
  crownoff_model       = "treeisonet_tls_boreal_crownoff_esegformer3D_128_15cm(GPU4GB)",
  conf_thresh          = 0.15,  # Python hardcodes 0.1; raised to 0.15 to suppress sapling phantoms
  max_isolated_dist    = 0.12,  # 3x voxel size; bridges stem/crown gaps without jumping trees
  k_graph              = 10L,   # matches Python stemCluster.shortestpath3D hardcoded k
  k_node               = 10L,   # matches Python stemCluster.create_node_graph() k; R default was 20
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
plot_ids  <- tree_locs[sqrt((lx - plot_ctr_x)^2 + (ly - plot_ctr_y)^2) <= r_max, TreeID]
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

# 3.5b Per-point spatial clip: remove distant TLS returns absorbed by CrownOff3D.
# The TLS scanner captures overstory returns out to 80m+ range. treeisonet() forms
# separate out-of-plot tree segments which Section 3.3 deletes by base position.
# But CrownOff3D can still absorb scattered distant returns into in-plot tree IDs;
# those points survive Section 3.3 because their tree BASE is within r_max.
# Clip to r_max + crown_extension_m to remove that residual scatter while retaining
# full crown geometry for in-plot trees near the boundary.
crown_extension_m <- 5.0   # m; headroom beyond plot edge for crown geometry
npts_before_clip <- npoints(las_segmented)
las_segmented <- filter_poi(las_segmented,
  sqrt((X - plot_ctr_x)^2 + (Y - plot_ctr_y)^2) <= (r_max + crown_extension_m))
message(sprintf("[3] Spatial clip (r <= %.1f m): %d -> %d pts (removed %d).",
                r_max + crown_extension_m, npts_before_clip, npoints(las_segmented),
                npts_before_clip - npoints(las_segmented)))

# Register treeisonet output columns as LAS extra bytes so writeLAS() persists them.
las_segmented <- lidR::add_lasattribute(las_segmented, las_segmented@data[["TreeID"]],   "TreeID",   "Tree ID")
las_segmented <- lidR::add_lasattribute(las_segmented, las_segmented@data[["StemCls"]],  "StemCls",  "Stem class")
las_segmented <- lidR::add_lasattribute(las_segmented, las_segmented@data[["TreeLocX"]], "TreeLocX", "Tree base X (m)")
las_segmented <- lidR::add_lasattribute(las_segmented, las_segmented@data[["TreeLocY"]], "TreeLocY", "Tree base Y (m)")

writeLAS(las_segmented, file.path(out_dir, "checkpoint_03b_segmented.laz"))

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
#las_segmented <- lidR::readLAS(file.path(out_dir, "checkpoint_03b_segmented.laz"))
#data.table::setDT(las_segmented@data)


# ============================================================================
# 3.6  UAV FUSION  (skip to Section 4 if no UAV data)
# Requires Section 2.5 registration: TLS must be in UTM before this runs.
# Run UAV treeisonet, LSAP stem matching, then merge TLS and UAV clouds.
# ============================================================================

# TLS-only fallback — run this block and skip to Section 4 if no UAV.
#las_fused <- las_segmented
#data.table::setDT(las_fused@data)
#las_fused@data[, Source := 1L]

# 3.6a  UAV treefiltering + treeisonet  -----------------------------------------
las_uav_raw <- lidR::readLAS(uav_input,
                 filter = paste("-keep_circle",
                                plot_utm_easting, plot_utm_northing,
                                plot_radius + plot_buffer + 5))
message(sprintf("[3.6a] UAV clip: %d pts.", npoints(las_uav_raw)))

# Decimate to 8 cm voxels (matches stemcls model resolution; reduces redundant pts).
npts_uav0   <- npoints(las_uav_raw)
las_uav_raw <- lidR::decimate_points(las_uav_raw, lidR::random_per_voxel(res = 0.08))
message(sprintf("[3.6a] UAV decimate: %d -> %d pts (%.1f%% retained).",
                npts_uav0, npoints(las_uav_raw), 100 * npoints(las_uav_raw) / npts_uav0))

las_uav_filt <- treeAIBoxR::treefiltering(
  las_uav_raw,
  model   = "treefiltering_uav_esegformer3D_128_12cm(GPU3GB)",
  sensor  = "uav",
  cuda    = TRUE,
  verbose = TRUE
)
las_uav_filt <- lidR::add_lasattribute(las_uav_filt,
  las_uav_filt@data[["TreeFilterLabel"]], "TreeFilterLabel", "Tree filter label")

las_uav_seg <- treeAIBoxR::treeisonet(
  las_uav_filt,
  sensor             = "uav",
  stemcls_model      = "treeisonet_uav_mixedwood_stemcls_esegformer3D_128_8cm(GPU3GB)",
  treeloc_model      = "treeisonet_uav_mixedwood_treeloc_esegformer3D_128_10cm(GPU3GB)",
  crownoff_model     = "treeisonet_uav_mixedwood_crownoff_esegformer3D_128_15cm(GPU4GB)",
  conf_thresh        = 0.15,
  max_isolated_dist  = 0.3,
  stemcls_min_points = 100L,
  cuda               = TRUE,
  verbose            = TRUE
)
las_uav_seg <- lidR::add_lasattribute(las_uav_seg, las_uav_seg@data[["TreeID"]],   "TreeID",   "Tree ID")
las_uav_seg <- lidR::add_lasattribute(las_uav_seg, las_uav_seg@data[["StemCls"]],  "StemCls",  "Stem class")
las_uav_seg <- lidR::add_lasattribute(las_uav_seg, las_uav_seg@data[["TreeLocX"]], "TreeLocX", "Tree base X (m)")
las_uav_seg <- lidR::add_lasattribute(las_uav_seg, las_uav_seg@data[["TreeLocY"]], "TreeLocY", "Tree base Y (m)")
data.table::setDT(las_uav_seg@data)
las_uav_seg@data[, Source := 2L]

# Post-filters mirroring TLS 3.3-3.5: plot boundary, minimum height, minimum points.
uav_r_max     <- plot_radius + plot_buffer
uav_tree_locs <- las_uav_seg@data[TreeID > 0L & !is.na(TreeLocX),
                   .(lx = TreeLocX[1L], ly = TreeLocY[1L]), by = TreeID]
uav_plot_ids  <- uav_tree_locs[sqrt((lx - plot_ctr_x)^2 + (ly - plot_ctr_y)^2) <= uav_r_max, TreeID]
las_uav_seg   <- lidR::filter_poi(las_uav_seg, TreeID %in% uav_plot_ids)

uav_heights   <- las_uav_seg@data[TreeID > 0L, .(height = max(Z) - min(Z)), by = TreeID]
uav_hgt_ids   <- uav_heights[height >= min_tree_height_m, TreeID]
las_uav_seg   <- lidR::filter_poi(las_uav_seg, TreeID %in% uav_hgt_ids)

uav_counts    <- las_uav_seg@data[TreeID > 0L, .N, by = TreeID]
uav_pt_ids    <- uav_counts[N >= min_tree_points, TreeID]
las_uav_seg   <- lidR::filter_poi(las_uav_seg, TreeID %in% uav_pt_ids)
message(sprintf("[3.6a] UAV after filters: %d trees.",
                las_uav_seg@data[TreeID > 0L & !is.na(TreeID), data.table::uniqueN(TreeID)]))

# Per-point spatial clip (mirrors TLS 3.5b): remove distant UAV returns that
# CrownOff3D absorbed into in-plot tree IDs (those points survive the base filter
# in Section 3.3 because their assigned tree base is within uav_r_max).
uav_npts_before_clip <- npoints(las_uav_seg)
las_uav_seg <- lidR::filter_poi(las_uav_seg,
  sqrt((X - plot_ctr_x)^2 + (Y - plot_ctr_y)^2) <= (uav_r_max + crown_extension_m))
message(sprintf("[3.6a] Spatial clip (r <= %.1f m): %d -> %d pts (removed %d).",
                uav_r_max + crown_extension_m, uav_npts_before_clip,
                npoints(las_uav_seg), uav_npts_before_clip - npoints(las_uav_seg)))

# Checkpoint written after filters so reloading always yields the filtered cloud.
lidR::writeLAS(las_uav_seg, file.path(out_dir, "checkpoint_04_uav_segmented.laz"))
# Recovery: las_uav_seg <- lidR::readLAS(file.path(out_dir, "checkpoint_04_uav_segmented.laz"))
#           data.table::setDT(las_uav_seg@data) ; las_uav_seg@data[, Source := 2L]
# QC: lidR::plot(las_uav_seg, color = "TreeID", pal = rainbow(50), bg = "white")

# 3.6b  Stem matching  (LSAP) ---------------------------------------------------
data.table::setDT(las_segmented@data)
las_segmented@data[, Source := 1L]

tls_stems <- las_segmented@data[!is.na(TreeID) & TreeID > 0L & !is.na(TreeLocX),
               .(X_stem = mean(TreeLocX), Y_stem = mean(TreeLocY), H_m = max(Z) - min(Z)),
               by = TreeID]
uav_stems <- las_uav_seg@data[!is.na(TreeID) & TreeID > 0L & !is.na(TreeLocX),
               .(X_stem = mean(TreeLocX), Y_stem = mean(TreeLocY), H_m = max(Z) - min(Z)),
               by = TreeID]
# uav_stems already has only filtered trees; retain the same height + plot guards
# for consistency with future runs that may reload from checkpoint.
uav_stems <- uav_stems[H_m >= min_tree_height_m]
uav_stems <- uav_stems[sqrt((X_stem - plot_ctr_x)^2 + (Y_stem - plot_ctr_y)^2) <= uav_r_max]
message(sprintf("[3.6b] TLS stems: %d   UAV stems: %d.", nrow(tls_stems), nrow(uav_stems)))

tls_std <- TreeMatching::standardize(as.data.frame(tls_stems),
             "X_stem", "Y_stem", "H_m", "m", crs = utm_crs, idname = "TreeID")
uav_std <- TreeMatching::standardize(as.data.frame(uav_stems),
             "X_stem", "Y_stem", "H_m", "m", crs = utm_crs, idname = "TreeID")
treemap <- TreeMatching::make_mapmatching(tls_std, uav_std,
             center = c(plot_ctr_x, plot_ctr_y), radius = uav_r_max)
treemap <- TreeMatching::match_trees(treemap, dxymax = 2.5, dzmax = 50, zrel = 0)
print(treemap)
# QC: plot(treemap, scale = 2)

mt      <- data.table::as.data.table(treemap$match_table)
matches <- mt[state == "Matched", .(tls_TreeID = id_inventory, uav_TreeID = id_measure)]
message(sprintf("[3.6b] Matched %d / %d TLS trees (%.0f%%).",
                nrow(matches), nrow(tls_stems), 100 * nrow(matches) / nrow(tls_stems)))

# 3.6c  Fuse  -------------------------------------------------------------------
# Matched UAV trees: relabel to TLS TreeID (points fuse into same tree object).
# Unmatched UAV trees: assign new IDs above the TLS ID range (UAV-only trees).
uav_data <- data.table::copy(las_uav_seg@data)
uav_data[TreeID %in% matches$uav_TreeID,
         TreeID := matches$tls_TreeID[match(TreeID, matches$uav_TreeID)]]

unmatched_uav_ids <- setdiff(uav_stems$TreeID, matches$uav_TreeID)
if (length(unmatched_uav_ids) > 0L) {
  max_id  <- max(las_segmented@data$TreeID, na.rm = TRUE)
  new_map <- setNames(seq_along(unmatched_uav_ids) + max_id,
                      as.character(unmatched_uav_ids))
  uav_data[TreeID %in% unmatched_uav_ids,
           TreeID := new_map[as.character(TreeID)]]
  message(sprintf("[3.6c] %d unmatched UAV trees \u2192 new IDs %d\u2013%d.",
                  length(unmatched_uav_ids), max_id + 1L,
                  max_id + length(unmatched_uav_ids)))
}
las_uav_seg@data <- uav_data

# Harmonise schemas: add missing columns with type-safe defaults, then zero-fill
# any NAs in standard LAS fields.  Must be done inline (not in a helper function)
# because [[<- on an S4 slot only persists when called directly on the slot.
.custom_las_cols <- c("TreeID", "TreeLocX", "TreeLocY", "StemCls",
                      "TreeFilterLabel", "conf", "Source")
.las_fill_val <- function(col, ref_dt) {
  cl <- class(ref_dt[[col]])[1]
  if (col %in% .custom_las_cols)          return(NA)
  if (cl == "logical")                    return(FALSE)
  if (cl %in% c("integer", "raw"))        return(0L)
  if (cl %in% c("numeric", "double"))     return(0.0)
  NA
}
# Add columns missing from UAV
for (.col in setdiff(names(las_segmented@data), names(las_uav_seg@data)))
  las_uav_seg@data[[.col]] <- .las_fill_val(.col, las_segmented@data)
# Add columns missing from TLS
for (.col in setdiff(names(las_uav_seg@data), names(las_segmented@data)))
  las_segmented@data[[.col]] <- .las_fill_val(.col, las_uav_seg@data)

# Zero-fill NAs in non-custom columns (e.g. ScannerChannel / ScanAngle in UAV).
for (.f in setdiff(names(las_uav_seg@data), .custom_las_cols)) {
  if (anyNA(las_uav_seg@data[[.f]])) {
    cl <- class(las_uav_seg@data[[.f]])[1]
    if (cl == "logical")
      las_uav_seg@data[is.na(get(.f)), (.f) := FALSE]
    else if (cl %in% c("integer", "raw"))
      las_uav_seg@data[is.na(get(.f)), (.f) := 0L]
    else if (cl %in% c("numeric", "double"))
      las_uav_seg@data[is.na(get(.f)), (.f) := 0.0]
  }
}
for (.f in setdiff(names(las_segmented@data), .custom_las_cols)) {
  if (anyNA(las_segmented@data[[.f]])) {
    cl <- class(las_segmented@data[[.f]])[1]
    if (cl == "logical")
      las_segmented@data[is.na(get(.f)), (.f) := FALSE]
    else if (cl %in% c("integer", "raw"))
      las_segmented@data[is.na(get(.f)), (.f) := 0L]
    else if (cl %in% c("numeric", "double"))
      las_segmented@data[is.na(get(.f)), (.f) := 0.0]
  }
}

# Align column order so lidR's rbind (positional matching) is safe.
all_cols <- union(names(las_segmented@data), names(las_uav_seg@data))
data.table::setcolorder(las_segmented@data, all_cols)
data.table::setcolorder(las_uav_seg@data,   all_cols)

las_fused <- rbind(las_segmented, las_uav_seg)
las_fused <- lidR::add_lasattribute(las_fused, las_fused@data[["Source"]],
               "Source", "Point source (1=TLS 2=UAV)")
lidR::writeLAS(las_fused, file.path(out_dir, "checkpoint_05_fused.laz"))
message(sprintf("[3.6c] Fused: %d TLS + %d UAV = %d pts, %d unique trees.",
                npoints(las_segmented), npoints(las_uav_seg), npoints(las_fused),
                las_fused@data[TreeID > 0L & !is.na(TreeID), data.table::uniqueN(TreeID)]))
# QC
lidR::plot(las_fused, color = "Source", pal = c("steelblue", "tomato"), bg = "white")







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
# Harmonise schemas before rbind (TLS has DownedLog; UAV branch-only pass does not).
for (.col in setdiff(names(las_tls_wc@data), names(las_uav_wc@data)))
  las_uav_wc@data[[.col]] <- .las_fill_val(.col, las_tls_wc@data)
for (.col in setdiff(names(las_uav_wc@data), names(las_tls_wc@data)))
  las_tls_wc@data[[.col]] <- .las_fill_val(.col, las_uav_wc@data)
.wc_all_cols <- union(names(las_tls_wc@data), names(las_uav_wc@data))
data.table::setcolorder(las_tls_wc@data, .wc_all_cols)
data.table::setcolorder(las_uav_wc@data, .wc_all_cols)
las_woodcls <- rbind(las_tls_wc, las_uav_wc)
las_woodcls <- treeAIBoxR::forest_components(las_woodcls)

# Register woodcls/forest_components output columns as LAS extra bytes so writeLAS() persists them.
las_woodcls <- lidR::add_lasattribute(las_woodcls, las_woodcls@data[["WoodLabel"]],      "WoodLabel",      "Wood label")
las_woodcls <- lidR::add_lasattribute(las_woodcls, las_woodcls@data[["DownedLog"]],      "DownedLog",      "Downed log flag")
las_woodcls <- lidR::add_lasattribute(las_woodcls, las_woodcls@data[["ComponentClass"]], "ComponentClass", "Forest component class")
writeLAS(las_woodcls, file.path(out_dir, "checkpoint_06_woodcls.laz"))
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
