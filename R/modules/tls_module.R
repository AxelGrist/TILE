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
library(Rcpp)

# Xeon w5-2545: 12 physical / 24 logical cores, 128 GB RAM. No competition
# for cycles on this box, so use all logical threads.
n_threads <- parallel::detectCores(logical = TRUE)   # 24
lidR::set_lidr_threads(n_threads)
if (requireNamespace("data.table", quietly = TRUE))
  data.table::setDTthreads(n_threads)
# BLAS/LAPACK thread cap is set in TILE/.Renviron (OMP/OPENBLAS/MKL_NUM_THREADS)
# and applied at R startup -- restart R if you change it.
options(lidR.progress = TRUE)

# ---- torch / CUDA auto-install ---------------------------------------------
# Installs torch backend once; picks highest supported CUDA build <= driver max.
local({
  if (!requireNamespace("torch", quietly = TRUE) ||
      isTRUE(torch::torch_is_installed())) return(invisible(NULL))

  supported <- c("12.8", "12.6")   # torch 0.17.0 max is cu128; update when 0.18+ adds cu129+

  hdr  <- tryCatch(system2("nvidia-smi", stdout = TRUE, stderr = FALSE), error = function(e) "")
  m    <- regmatches(hdr, regexpr("[0-9]+\\.[0-9]+", hdr[grep("CUDA Version", hdr)[1L]]))
  dnum <- if (length(m) == 1L) as.integer(sub("\\.", "", m)) else 0L  # e.g. 131

  kind <- Filter(function(v) as.integer(gsub("\\.", "", v)) <= dnum, supported)
  kind <- if (length(kind)) kind[[1L]] else "cpu"

  message(sprintf("[torch] Installing LibTorch %s build.", kind))
  Sys.setenv(CUDA = kind)
  torch::install_torch()
})


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

  # --- Section 5: Per-tree QC PNGs -----------------------------------
  # Save one PNG per TreeID under <out_dir>/tree_qc/. When RGB is available
  # in the LAS, each PNG is a 2x2 grid:
  #   top-row    : XZ + YZ in true RGB
  #   bottom-row : XZ + YZ classified wood (brown) vs foliage (green)
  #                using the Excess-Green index ExG = 2G - R - B.
  # When RGB is absent / all zero, falls back to the legacy 1x2 cyan/orange
  # layout. Useful for QC-ing multi-stem clumps and wood/foliage purity.
  tree_qc_enable         = TRUE,
  tree_qc_subdir         = "tree_qc",
  tree_qc_min_points     = 50L,    # skip TreeIDs sparser than this
  tree_qc_xy_halfwidth   = 3.0,    # m; half-width of XZ/YZ panel
  tree_qc_top_n          = NA,     # NA = all; set to e.g. 30 to limit to
                                   # the largest-DBH TreeIDs (faster)
  tree_qc_use_rgb        = TRUE,   # auto-detect RGB; FALSE to force the
                                   # cyan/orange legacy layout.
  # Wood/foliage classifier backend for the bottom row of the QC plot:
  #   "pca"       : pure-R local-PCA linearity (no extra deps)
  #   "treeaibox" : R-torch port of NRCan TreeAIBox's vox3DSegFormer
  #                 (Xi, Hopkinson & Chasmer 2018; Xi, Chasmer & Hopkinson
  #                 2023). Requires the treeAIBoxR package, the R `torch`
  #                 backend installed (torch::install_torch() once), and
  #                 a .pth weights file + matching .json config from
  #                 https://github.com/NRCan/TreeAIBox/releases/tag/v1.0.
  tree_qc_wf_method      = "treeaibox",
  # --- pca backend knobs (always read; ignored when method != "pca") ---
  # PCA: per point, take its k nearest neighbours, build the 3x3
  # covariance, and compute linearity L = (l1 - l2) / l1. Linear
  # neighbourhoods (high L) are wood/branches; scattered neighbourhoods
  # (low L) are foliage. Geometric baseline from Belton 2013 / Xi 2018.
  tree_qc_wf_k           = 10L,    # k neighbours for local PCA (10-20 ok)
  tree_qc_wf_linearity   = 0.85,   # linearity threshold; >thr -> wood
  tree_qc_wf_max_pts     = 60000L, # subsample if a tree exceeds this
                                   # (kNN cost scales ~ n log n)
  # --- treeaibox backend knobs (only used when method == "treeaibox") --
  # Preferred: pass a model name and it auto-downloads on first use.
  # Run treeAIBoxR::treeaibox_model_zoo() to see all available names.
  tree_qc_treeaibox_model   = "woodcls_branch_tls_segformer3D_112_4cm(GPU2GBDistilled)",
  # Legacy: set these only when pointing at manually downloaded files.
  tree_qc_treeaibox_weights = NA_character_,  # path to .pth file
  tree_qc_treeaibox_config  = NA_character_,  # path to matching .json
  tree_qc_treeaibox_device  = "auto",         # "cuda" | "cpu" | "auto"
  tree_qc_wood_color     = "#A0522D",  # sienna
  tree_qc_foliage_color  = "#2E8B57",  # sea green
  # --- Section 3.3: Stem base detection -----------------------------------
  # Shape-aware: keep slice points that are vertical AND planar (trunk-like),
  # then 3D DBSCAN-cluster them. 3D clustering separates leaning/close stems
  # that 2D density would merge.
  #
  # Multi-slice union: instead of clustering one [zmin, zmax] band, cluster
  # several narrow bands independently and union the centroids. Multi-stem
  # clumps that braid together at one height often separate cleanly at
  # another (root flare vs mid-bole). Set base_slices = NULL to fall back
  # to the single [base_zmin, base_zmax] band.
  #
  # IMPORTANT (plot 311 calibration): low slices (<3 m) sit inside a dense
  # foliage skirt that daisy-chains adjacent stems through interlocking
  # branches. The 5-10 m mid-bole zone shows clean separated trunks. We
  # detect bases there, then Section 6 recomputes XY from the [0.10, 0.50]
  # m base slice for any TreeID that has enough points there (otherwise
  # falls back to BH XY). This gives clump separation at altitude with
  # base-accurate XY where possible.
  base_zmin            = 3.0,    # m; above foliage skirt
  base_zmax            = 10.0,   # m; below upper-crown foliage occlusion
  base_slices          = list(   # list of c(zmin, zmax) pairs; NULL = single
    c(3.0, 5.0),                 #   just above skirt
    c(5.0, 7.0),                 #   cleanest separation zone
    c(7.0, 10.0)                 #   upper-bole backup for occluded mid-bole
  ),
  base_res             = 0.04,   # m; 3D DBSCAN eps -- tight enough to keep
                                 #    sapling clusters and close thin stems
                                 #    in separate clusters
  base_min_vert        = 0.85,   # strict: clean trunks at 3-10 m are bare
                                 #         and very vertical; loose values
                                 #         let foliage residuals back in.
  base_min_plan        = 0.55,
  base_min_cluster     = 12,     # slightly relaxed since fewer points per
                                 # slice at height (vs ground-level slice).
  base_merge_eps       = 0.30,   # m; fold cross-slice twins. Mid-bole
                                 # centroids of a single leaning stem can
                                 # drift up to ~20-25 cm between slices,
                                 # so this floor keeps them folded into
                                 # one base. Real neighbor stems are
                                 # generally >0.5 m apart so this doesn't
                                 # over-merge.
  # Raster fallback (uncomment + swap call below if geom misbehaves):
  # base_density_q       = 0.99,
  # base_merge_eps_raster= 0.15,

  # RANSAC cylinder validation (per-cluster, per-slice). After DBSCAN
  # clusters survive the verticality+planarity filter and the min-cluster
  # gate, fit a vertical-axis cylinder (i.e. a 2D circle to the XY
  # projection) to each cluster via RANSAC. Reject clusters whose:
  #   * inlier ratio is too low (foliage tufts / branch stubs / leaning
  #     fragments fail the cylinder hypothesis even when they pass the
  #     eigen filter)
  #   * fitted radius falls outside a plausible trunk-radius window
  #   * vertical extent is too small a fraction of the slice thickness
  #     (real trunks span the slice; foliage clusters are thin pancakes)
  # Approach motivated by Zhu et al. 2024 (Forests 15:136), where a
  # RANSAC cylinder gate eliminated ~50 false-positive shrubs that CSP's
  # eigen filter alone could not.
  base_cylinder_validate     = TRUE,
  base_cyl_min_inlier_ratio  = 0.55,  # frac of cluster XY within tol of circle
  base_cyl_inlier_tol        = 0.04,  # m; |dist_to_circle - r| <= tol -> inlier
  base_cyl_radius_min        = 0.015, # m; ~3 cm DBH lower bound
  base_cyl_radius_max        = 0.30,  # m; ~60 cm DBH upper bound
  base_cyl_min_vert_frac     = 0.40,  # frac of slice thickness the cluster
                                      # must span vertically
  base_cyl_iters             = 80L,   # RANSAC iterations per cluster
  base_cyl_seed              = 1L,    # RNG seed for repeatability

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

  # --- Section 3.4: Segmentation method ----------------------------------
  # Pluggable: pick which routine assigns TreeID to every point given the
  # set of validated bases. All three accept the same `bases` data.frame
  # so they are directly A/B-comparable on identical seeds.
  #   "csp"          : CspStandSegmentation voxel-graph Dijkstra (Tao 2015).
  #                    Strong on sparse, structurally simple stands.
  #   "nearest_base" : XY Voronoi from base centroids. Fastest, no graph.
  #                    Useful as a sanity-check baseline -- if CSP/SHS
  #                    don't beat it, the bases are doing all the work.
  #   "shs"          : Full SHS (Zhu 2024, Forests 15:136). Builds a 3D KNN
  #                    adjacency graph over the canopy point cloud and runs
  #                    multi-source binary-heap Dijkstra (Rcpp) from each
  #                    trunk centroid. Each point assigned to the trunk with
  #                    minimum geodesic distance through actual point-cloud
  #                    connectivity -- correctly handles overhanging crowns
  #                    and air gaps that 3D-Euclidean approaches mis-route.
  #                    Trunk slice (Z <= shs_trunk_zmax) handled separately
  #                    by nearest-base XY (matches paper Section 2.4 step 1).
  #   "treeiso"      : Native R/Rcpp port of CloudCompare's qTreeIso plugin
  #                    (Xi & Hopkinson 2022, doi:10.3390/rs14236116).
  #                    Three-stage cut-pursuit pipeline operating directly
  #                    on canopy XYZ; ignores `bases` and produces its own
  #                    tree count.
  seg_method           = "treeisonet",

  # --- TreeFiltering knobs (optional pre-filter for treeisonet) ------------
  # TreeFiltering runs a supervised DL classifier (ESegFormer3D) to separate
  # overstory (class 2) from understory/ground (class 1) before TreeisoNet.
  # When enabled (treeisonet_treefilter_model is not NA), only overstory
  # points are passed to StemCls / TreeLoc. Mirrors the GUI workflow where
  # TreeFiltering is applied first and subsequent steps use the treefilter
  # scalar field to mask the input cloud.
  #
  # Model names must exist in treeaibox_model_zoo(); auto-downloaded on first use.
  # Recommended:
  #   TLS:  "treefiltering_tls_esegformer3D_128_8cm(GPU3GB)"
  #   ALS:  "treefiltering_als_esegformer3D_128_50cm(GPU3GB)"   (50 cm regular)
  #         "treefiltering_als_esegformer3D_128_80cm(GPU3GB)"   (80 cm mountainous)
  #         "treefiltering_als_esegformer3D_128_15cm(GPU3GB)"   (15 cm wellsite)
  #   UAV:  "treefiltering_uav_esegformer3D_128_12cm(GPU3GB)"
  treeisonet_treefilter_model   = "treefiltering_tls_esegformer3D_128_8cm(GPU3GB)",
  treeisonet_treefilter_device  = "auto",         # "auto" | "cuda" | "cpu"
  # if_bottom_only: use 2D XY-only sliding blocks (TRUE) or full 3D (FALSE).
  # TRUE:  classify each XY column using only the bottom 10.24 m (trunk region),
  #        then auto-mark ALL points above 10.24 m as overstory.  Correct for
  #        TLS when trees exceed one block height (~10 m); avoids sparse treetop
  #        blocks being misclassified as understory by the 3-D sliding window.
  # FALSE: full 3-D sliding; upper crown blocks are sparse and may appear as
  #        understory, causing a hard cutoff at the top of the last block.
  treeisonet_treefilter_bottom_only = TRUE,

  # --- TreeisoNet knobs (used when seg_method == "treeisonet") -------------
  # Deep-learning individual tree segmentation pipeline (Xi et al. 2023):
  #
  #   TLS boreal / UAV mixedwood:
  #     StemCls (ESegformer3D) → TreeLoc (Detection) → shortestpath3D (graph)
  #     Set treeisonet_stemcls_model + treeisonet_treeloc_model.
  #
  #   ALS reclamation:
  #     TreeLoc (Detection) → TreeOff (Regression) → mergeshift (kNN)
  #     Set treeisonet_treeloc_model + treeisonet_treeoff_model; leave
  #     treeisonet_stemcls_model = NA_character_ to activate the ALS path.
  #
  # Model names must exist in treeaibox_model_zoo(); auto-downloaded on first use.
  treeisonet_stemcls_model    = "treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)",
  treeisonet_treeloc_model    = "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)",
  treeisonet_treeoff_model    = NA_character_,   # set for ALS reclamation path
  treeisonet_device           = "auto",  # "auto" | "cuda" | "cpu"

  # Custom voxel resolution (NA = use resolution embedded in model config)
  treeisonet_vox_xy           = NA_real_,  # m; overrides model XY voxel size
  treeisonet_vox_z            = NA_real_,  # m; overrides model Z voxel size

  # StemCls post-processing — "Remove small clusters" in the GUI
  treeisonet_stemcls_remove_small = TRUE,  # remove isolated small stem clusters
  treeisonet_stemcls_max_gap      = 3.0,   # m; connectivity gap tolerance
  treeisonet_stemcls_min_points   = 100L,  # min cluster size to keep as stem

  # TreeLoc detection tuning
  # TLS/UAV path (if_stem=TRUE, treeloc-cutoff-content panel in GUI):
  treeisonet_cutoff_thresh    = 0.3,    # fraction of per-tile stem height to use
  treeisonet_cut_grid_res     = 5.0,    # m; tile width for height normalisation
  treeisonet_base_radius_m    = 0.2,    # m; search radius when snapping base XYZ
  # ALS path (if_stem=FALSE, treeloc-default-content panel in GUI):
  treeisonet_conf_thresh      = 0.3,    # min per-point confidence for peak map
  treeisonet_nms_thresh_xy    = 0.5,    # NMS suppression: multiplied by sum of radii
  treeisonet_treeloc_max_gap  = 0.3,    # m; max gap in postPeakExtraction kNN graph
  treeisonet_treeloc_K        = 5L,     # kNN k for postPeakExtraction grouping

  # shortestpath3D graph tuning (TLS / UAV path only)
  treeisonet_min_res          = 0.06,   # m; voxel edge for graph decimation
  treeisonet_max_isolated_dist = 0.3,   # m; max edge length in kNN graph
  treeisonet_k_graph          = 10L,    # kNN k for point-level graph edges
  treeisonet_k_node           = 20L,    # kNN k for component-centroid graph

  # --- TreeIso knobs (used when seg_method == "treeiso") ------------------
  # Native R/Rcpp port of the CloudCompare qTreeIso plugin (Xi & Hopkinson
  # 2022, doi:10.3390/rs14236116). Runs the original 3-stage cut-pursuit
  # pipeline directly on the canopy XYZ; ignores `bases` and produces its
  # own tree count (set tree_min_height/tree_min_dbh to 0 to keep all
  # treeiso outputs through validation).
  treeiso_K1                  = 5L,    # stage 1: kNN
  treeiso_lambda1             = 1.0,   # stage 1: cut-pursuit reg strength
  treeiso_dec_r1              = 0.05,  # stage 1: voxel decimation (m)
  treeiso_K2                  = 20L,   # stage 2: kNN over cluster centroids
  treeiso_lambda2             = 20.0,  # stage 2: cut-pursuit reg strength
  treeiso_max_gap             = 2.0,   # stage 2: max 3D gap between segs (m)
  treeiso_dec_r2              = 0.10,  # stage 2: voxel decimation (m)
  treeiso_K3                  = 20L,   # stage 3: kNN for refinement
  treeiso_rel_h_len_r         = 0.5,   # stage 3: rel-height/length threshold
  treeiso_vert_w              = 0.5,   # stage 3: vertical-overlap weight
  treeiso_threads             = 6L,    # OpenMP threads
  treeiso_verbose             = FALSE,

  # --- SHS knobs (used when seg_method == "shs") --------------------------
  shs_trunk_zmax       = 4.0,    # m; trunk-layer / canopy-layer split
  shs_canopy_voxel     = 0.10,   # m; canopy thinning before graph build.
                                 # 14M -> ~1-2M nodes keeps Dijkstra under
                                 # 60 s and graph memory <2 GB. Final labels
                                 # are propagated back to the full cloud
                                 # via 3D nearest-neighbor lookup.
  shs_knn_k            = 8L,     # KNN k for adjacency graph
  shs_knn_max_dist     = 0.50,   # m; edges longer than this are dropped
                                 # (forces routing through actual point-
                                 # cloud connectivity rather than across
                                 # air gaps -- the whole point of geodesic).

  # --- CSP cost-segmentation knobs (used when seg_method == "csp") --------
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

  # --- Section 6: Forest inventory + tree filter --------------------------
  inv_slice_min        = 0.3,    # m; bottom of taper band
  inv_slice_max        = 4.0,    # m; top of taper band
  inv_increment        = 0.1,    # m; vertical step
  inv_width            = 0.05,   # m; slice thickness for circle fit
  inv_max_dbh          = 1.0,    # m; reject impossible stems
  inv_n_cores          = parallel::detectCores(logical = TRUE),
  tree_min_dbh         = 0.0,    # TEMP for TreeIso QC: was 0.04. Field min
                                 # for plot 311 = 5.8 cm. Restore to 0.04
                                 # once TreeIso output is validated.
  tree_min_height      = 0.0,    # TEMP for TreeIso QC: was 4.0. Field min
                                 # for plot 311 = 9.0 m. Restore to 4.0
                                 # once TreeIso output is validated.

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

  # --- Section 4: WoodCls / leaf-wood separation --------------------------
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

  # --- Section 4: WoodCls (DL classifier) --------------------------------
  # WoodCls runs a supervised DL binary classifier (ESegFormer3D) per point
  # to separate wood (branches + trunk, label=2) from foliage (label=1).
  # Output is written to las@data$WoodLabel.
  # The QSM pipeline (Section 9) uses WoodLabel==2 points as input.
  # Set woodcls_model = NA_character_ to skip (QSM will not run either).
  #
  # Recommended models:
  #   woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)  ← 2 GB GPU, 2.5 cm vox
  #   woodcls_branch_tls_segformer3D_112_4cm(GPU2GBDistilled) ← distilled, faster
  #   woodcls_stem_tls_esegformer3D_128_4cm(GPU3GB)
  woodcls_model  = "woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)",
  woodcls_device = "auto",   # "auto" | "cuda" | "cpu"

  # --- Section 4b: Downed log detection (Xi class 4) ----------------------
  # Xi 2023 method: run WoodCls with if_bottom_only=TRUE (2D XY voxelisation)
  # on near-ground points. In 2D mode the model sees full vertical columns in
  # XY, which better captures horizontal log geometry than 3D blocks. The
  # output is then post-filtered by a PCA inclination-angle threshold:
  # Xi 2023: ">60 deg from vertical clearly demarcates the leaning tree layer
  # from the downed woody debris layer". Z is already normalised (Section 2.1).
  downed_log_enable    = TRUE,   # FALSE to skip and leave class 4 unpopulated
  downed_z_max         = 1.5,    # m above ground; only points below this are tested
  downed_tilt_min_deg  = 60.0,   # PCA primary-axis tilt from vertical (degrees)
                                  # Xi 2023: >60 deg threshold

  # --- Section 9: QSM (TreeAIBox) -----------------------------------------
  # Builds a Quantitative Structure Model per tree using the applyQSM
  # pipeline (Xi et al.; cut-pursuit over-segmentation + Dijkstra skeleton +
  # algebraic circle-fit radii). Requires treeisoR >= 0.2.0 (cut_pursuit_segment).
  # Set qsm_tree_ids = NA to run on all trees with TreeID > 0.
  qsm_tree_ids           = NA_integer_,   # integer vector or NA = all trees
  # cut-pursuit over-segmentation
  qsm_K_stem             = 20L,     # kNN for stem seg (more neighbours → smoother)
  qsm_reg_stem           = 5.0,     # regularisation for stem (high → fewer segs)
  qsm_K_branch           = 3L,      # kNN for branch seg
  qsm_reg_branch         = 0.01,    # regularisation for branches (low → more segs)
  # skeleton graph
  qsm_k_neighbors        = 6L,      # kNN for segment connectivity graph
  qsm_max_graph_distance = 40,      # max Dijkstra path length (graph edge units)
  qsm_max_conn_dist      = 0.03,    # m; cross-segment point adjacency radius
  qsm_occlusion_cutoff   = 0.4,     # m; edges beyond this are cut as occluded
  qsm_min_pts_clean      = 5L,      # min pts in branch tip to keep that branch
  qsm_min_radius_m       = 0.04,    # m; minimum enforced cylinder radius
  qsm_threads            = 1L,      # OpenMP threads for cut-pursuit
  qsm_out_dir            = NA_character_,  # NA = use out_dir from pipeline

  # --- Section 8: Field-data validation -----------------------------------
  # For each field tree, find ALL TLS candidates inside a buffer radius,
  # then pick the candidate with the lowest weighted score across
  # available metrics (xy / dbh / height / azimuth / distance).
  # Matched field info is written back as columns on `inv` (prefix `f_`).
  # Setting field_xlsx_path = NA disables Section 8.
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

  field_match_buffer   = 2.0,    # m; radius around each field tree to
                                 # gather TLS candidates from `inv`. 2.0 m
                                 # is generous enough to catch persistent
                                 # multi-stem clumps where the field XY
                                 # was recorded for a stem that merged
                                 # into a neighbor at TLS base height.
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
    lbl <- forest_component_labels(ds)
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
                  pch = 19, bty = "n", cex = 1.1)
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
# 1. READ + INSPECT
# ============================================================================

las <- readLAS(tls_input, select = "xyzicrnRGB")  # keep core attrs + RGB
if (is.empty(las)) stop("Input LAS is empty / failed to read.")

las_check(las)
print(las)

# LAS field inventory -- which extra signals are actually populated?
# Used to decide whether the optional Intensity / NumberOfReturns / RGB
# gates in the leaf-wood separation block (Section 4) are worth enabling.
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
# Keep ground-classified points regardless of z_min: after normalize_height
# they sit at Z ~ 0, which is below the z_min noise floor (0.2 m default).
las <- filter_poi(las, Classification == 2L | (Z >= tls_params$z_min & Z <= tls_params$z_max))

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
#
# Skipped when seg_method='treeisonet' with a treefilter model set:
# TreeFilter (ESegFormer3D) classifies overstory vs shrub/ground directly
# and is strictly better than this hand-tuned eigen-geometry rule.
.use_treefilter <- identical(tolower(tls_params$seg_method), "treeisonet") &&
  !is.null(tls_params$treeisonet_treefilter_model) &&
  !is.na(tls_params$treeisonet_treefilter_model) &&
  nzchar(tls_params$treeisonet_treefilter_model)
if (.use_treefilter) {
  message("3.2 Shrub filter: skipped (TreeFilter model handles overstory/shrub separation).")
}
npts0 <- npoints(las)
if (!.use_treefilter) {
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
} # end !.use_treefilter

# 3.3 Stem base detection ----------------------------------------------------
# Shape-aware base finder. Local fast variant of
# CspStandSegmentation::find_base_coordinates_geom() that reuses the
# eigen columns we already computed in 3.1. Skipped when TreeisoNet runs
# its own TreeLoc model internally (see .use_treefilter guard below).
#
# CspStandSegmentation::find_base_coordinates_geom() that reuses the
# eigen columns we already computed in 3.1. Otherwise identical logic
# (CspStandSegmentation 0.2.0).
#
# Multi-slice union: when `slices` is a list of c(zmin, zmax) pairs, run
# the per-slice DBSCAN cluster -> centroid pipeline once per slice and
# union the centroids before the cross-slice merge_eps fold. Multi-stem
# clumps that braid together at one height often separate cleanly at
# another. When `slices` is NULL, falls back to a single [zmin, zmax]
# slice (legacy behavior).
find_bases_geom_fast <- function(las, zmin, zmax, res,
                                 min_verticality, min_planarity,
                                 min_cluster_size, merge_eps = 0.20,
                                 slices = NULL,
                                 cylinder_validate    = FALSE,
                                 cyl_min_inlier_ratio = 0.55,
                                 cyl_inlier_tol       = 0.04,
                                 cyl_radius_min       = 0.015,
                                 cyl_radius_max       = 0.30,
                                 cyl_min_vert_frac    = 0.40,
                                 cyl_iters            = 80L,
                                 cyl_seed             = 1L) {

  # Fit a 2D circle to three non-collinear points; return c(cx, cy, r) or
  # NULL if the points are (near-)collinear. Used as the RANSAC sample
  # hypothesis for cylinder validation (vertical-axis cylinder == circle
  # in XY).
  circle_from_3 <- function(p) {
    ax <- p[1, 1]; ay <- p[1, 2]
    bx <- p[2, 1]; by <- p[2, 2]
    cx_ <- p[3, 1]; cy_ <- p[3, 2]
    d <- 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if (abs(d) < 1e-9) return(NULL)
    ux <- ((ax^2 + ay^2) * (by - cy_) +
           (bx^2 + by^2) * (cy_ - ay) +
           (cx_^2 + cy_^2) * (ay - by)) / d
    uy <- ((ax^2 + ay^2) * (cx_ - bx) +
           (bx^2 + by^2) * (ax - cx_) +
           (cx_^2 + cy_^2) * (bx - ax)) / d
    r <- sqrt((ax - ux)^2 + (ay - uy)^2)
    c(ux, uy, r)
  }

  # RANSAC vertical-cylinder gate. Returns TRUE if the cluster looks like
  # a piece of a trunk: enough XY points lie on a circle of plausible
  # radius, AND the cluster spans enough vertical extent.
  cylinder_ok <- function(xyz, z1, z2) {
    n <- nrow(xyz)
    if (n < 6L) return(FALSE)
    # Vertical extent gate (fast reject): foliage tufts are thin pancakes.
    z_extent <- diff(range(xyz[, 3]))
    if (z_extent < cyl_min_vert_frac * (z2 - z1)) return(FALSE)
    xy <- xyz[, 1:2, drop = FALSE]
    best_inliers <- 0L
    best_r <- NA_real_
    for (it in seq_len(cyl_iters)) {
      idx <- sample.int(n, 3L)
      cir <- circle_from_3(xy[idx, , drop = FALSE])
      if (is.null(cir)) next
      r <- cir[3]
      if (r < cyl_radius_min || r > cyl_radius_max) next
      d <- sqrt((xy[, 1] - cir[1])^2 + (xy[, 2] - cir[2])^2)
      ninl <- sum(abs(d - r) <= cyl_inlier_tol)
      if (ninl > best_inliers) {
        best_inliers <- ninl
        best_r <- r
      }
    }
    if (best_inliers == 0L) return(FALSE)
    if (is.na(best_r) ||
        best_r < cyl_radius_min || best_r > cyl_radius_max) return(FALSE)
    (best_inliers / n) >= cyl_min_inlier_ratio
  }

  # Per-slice helper: returns a data.frame(X, Y, Z) of centroids, or
  # NULL if the slice is empty / no clusters survive.
  one_slice <- function(z1, z2) {
    sl <- lidR::filter_poi(las, Classification != 2L & Z > z1 & Z < z2)
    if (lidR::is.empty(sl)) return(NULL)
    if (!all(c("Verticality", "Planarity") %in% names(sl@data)))
      sl <- CspStandSegmentation::add_geometry(sl)
    sl <- lidR::filter_poi(sl,
                           Planarity   > min_planarity &
                           Verticality > min_verticality)
    if (lidR::is.empty(sl)) return(NULL)
    cl <- dbscan::dbscan(sl@data[, 1:3], eps = res, minPts = 1)$cluster
    keep <- as.integer(names(table(cl))[table(cl) > min_cluster_size])
    if (!length(keep)) return(NULL)
    sl <- lidR::filter_poi(sl, cl %in% keep)
    cl <- cl[cl %in% keep]
    # RANSAC cylinder validation per cluster (opt-in). Drops clusters
    # that pass the eigen filter but don't actually look like a trunk
    # piece (foliage tufts, charred branch stubs, leaning fragments).
    if (isTRUE(cylinder_validate)) {
      set.seed(cyl_seed)
      pts <- as.matrix(sl@data[, 1:3])
      n_pre <- length(keep)
      keep_ok <- vapply(keep, function(k)
        cylinder_ok(pts[cl == k, , drop = FALSE], z1, z2),
        logical(1))
      keep <- keep[keep_ok]
      if (!length(keep)) return(NULL)
      sl <- lidR::filter_poi(sl, cl %in% keep)
      cl <- cl[cl %in% keep]
      message(sprintf("  RANSAC cylinder gate [%.2f-%.2f]: %d / %d clusters passed.",
                      z1, z2, length(keep), n_pre))
    }
    xy <- aggregate(sl@data[, 1:2], by = list(cl), mean)
    # NOTE: base Z is intentionally set to ground level (not min Z of the
    # cluster) so CSP's voxel graph can always reach the seed. Bases at
    # mid-bole height (3-10 m) often land in occluded gaps between voxels
    # and produce "Invalid vertex names" -> unreachable warnings, leaving
    # those TreeIDs with zero segmented points. With Z = ground, every
    # seed lives in the dense ground layer where Dijkstra can spread out.
    # XY is still the mid-bole centroid (good clump separation); Section
    # 4 then recomputes XY from the [0.10, 0.50] m slice for accuracy.
    data.frame(X = xy[, 2], Y = xy[, 3], Z = 0.5)
  }

  if (is.null(slices) || !length(slices)) {
    bases <- one_slice(zmin, zmax)
    if (is.null(bases))
      stop("No clusters survived in [zmin, zmax].")
    n_per <- nrow(bases)
    message(sprintf("Base detection (single slice [%.2f, %.2f]): %d centroids.",
                    zmin, zmax, n_per))
  } else {
    parts <- lapply(slices, function(s) one_slice(s[1], s[2]))
    n_per <- vapply(parts, function(p) if (is.null(p)) 0L else nrow(p), integer(1))
    message(sprintf("Base detection (multi-slice union): %s -> %d centroids before merge.",
                    paste(sprintf("[%.2f-%.2f]=%d",
                                  vapply(slices, `[`, numeric(1), 1),
                                  vapply(slices, `[`, numeric(1), 2),
                                  n_per), collapse = ", "),
                    sum(n_per)))
    parts <- parts[!vapply(parts, is.null, logical(1))]
    if (!length(parts))
      stop("No clusters survived in any slice.")
    bases <- do.call(rbind, parts)
  }

  # Merge near-duplicate bases (DBSCAN can split one trunk into multiple
  # vertical-planar fragments at slightly different heights; multi-slice
  # mode also produces one centroid per stem per slice). Collapse any
  # bases within `merge_eps` (XY) into a single mean centroid. Without
  # this step, CSP routes each duplicate seed's voxels to whichever stem
  # wins the Dijkstra tiebreak, absorbing entire neighbor trunks.
  if (nrow(bases) > 1 && merge_eps > 0) {
    nb_pre <- nrow(bases)
    mc <- dbscan::dbscan(bases[, c("X", "Y")], eps = merge_eps, minPts = 1)$cluster
    bases <- aggregate(bases, by = list(mc), mean)[, -1]
    message(sprintf("Base merge (eps=%.2f m): %d -> %d centroids.",
                    merge_eps, nb_pre, nrow(bases)))
  }
  bases$TreeID <- seq_len(nrow(bases))
  bases
}

if (.use_treefilter) {
  # TreeisoNet runs TreeLoc internally — base detection is skipped.
  # Create an empty stub so the switch dispatcher has a `bases` variable.
  bases <- data.frame(X = numeric(0), Y = numeric(0), Z = numeric(0),
                      TreeID = integer(0))
  message("3.3 Base detection: skipped (TreeLoc model handles base detection internally).")
} else {
bases <- find_bases_geom_fast(las,
  zmin             = tls_params$base_zmin,
  zmax             = tls_params$base_zmax,
  res              = tls_params$base_res,
  min_verticality  = tls_params$base_min_vert,
  min_planarity    = tls_params$base_min_plan,
  min_cluster_size = tls_params$base_min_cluster,
  merge_eps        = tls_params$base_merge_eps,
  slices           = tls_params$base_slices,
  cylinder_validate    = tls_params$base_cylinder_validate,
  cyl_min_inlier_ratio = tls_params$base_cyl_min_inlier_ratio,
  cyl_inlier_tol       = tls_params$base_cyl_inlier_tol,
  cyl_radius_min       = tls_params$base_cyl_radius_min,
  cyl_radius_max       = tls_params$base_cyl_radius_max,
  cyl_min_vert_frac    = tls_params$base_cyl_min_vert_frac,
  cyl_iters            = tls_params$base_cyl_iters,
  cyl_seed             = tls_params$base_cyl_seed
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
} # end !.use_treefilter (base detection block)

# 3.4 Segmentation -----------------------------------------------------------
# Dispatcher: assigns TreeID to every point. All three methods take the
# same validated `bases` and return a LAS with @data$TreeID populated.

# Pure XY Voronoi from base centroids -- fastest, no graph. Useful as a
# baseline: if a more expensive method does not improve over this, the
# bases (not the routing) are doing all the work.
segment_trees_nearest_base <- function(las, bases) {
  if (!requireNamespace("RANN", quietly = TRUE))
    stop("seg_method='nearest_base' requires the RANN package.")
  pts_xy <- as.matrix(las@data[, .(X, Y)])
  base_xy <- as.matrix(bases[, c("X", "Y")])
  nn <- RANN::nn2(base_xy, pts_xy, k = 1L)
  las@data$TreeID <- bases$TreeID[nn$nn.idx[, 1]]
  message(sprintf("seg_method='nearest_base': %d points -> %d trees.",
                  nrow(las@data), nrow(bases)))
  las
}

# SHS-flavored hierarchical segmentation (Zhu 2024, Forests 15:136). Full
# implementation:
#   1. Trunk slice (Z <= trunk_zmax): assign each point to its nearest
#      base centroid in XY (paper Section 2.4 step 1).
#   2. Canopy slice (Z > trunk_zmax):
#      a. Optionally voxel-thin to keep the graph tractable. Final labels
#         get propagated back to the full cloud via 3D nearest-neighbor.
#      b. Build a 3D KNN adjacency graph (RANN), drop edges > knn_max_dist
#         so paths must traverse real point-cloud connectivity.
#      c. Multi-source binary-heap Dijkstra (Rcpp) from a virtual super-
#         source connected with weight 0 to all trunk centroids. Each
#         canopy node's parent chain ends at the trunk centroid that
#         achieves minimum geodesic distance -- this is the algorithm in
#         the paper (Algorithm 1).
#      d. Propagate labels from thinned canopy back to full canopy via
#         3D nearest-neighbor.
#
# The Rcpp Dijkstra kernel is compiled once on first use via cppFunction.
# This requires Rtools (Windows) or a system C++ toolchain.
.shs_dijkstra_compiled <- new.env(parent = emptyenv())
.shs_dijkstra_compiled$fn <- NULL

.shs_compile <- function() {
  if (!is.null(.shs_dijkstra_compiled$fn)) return(invisible(NULL))
  if (!requireNamespace("Rcpp", quietly = TRUE))
    stop("seg_method='shs' requires the Rcpp package.")
  message("Compiling SHS multi-source Dijkstra kernel (Rcpp)...")
  Rcpp::cppFunction(depends = "", plugins = "cpp11", code = '
    #include <Rcpp.h>
    #include <vector>
    #include <queue>
    #include <utility>
    #include <limits>
    using namespace Rcpp;

    // Multi-source binary-heap Dijkstra over a CSR adjacency graph.
    //   adj_idx, adj_w : flat neighbor arrays (length = sum of degrees)
    //   adj_ptr        : length n+1; node i has neighbors adj_idx[adj_ptr[i]..adj_ptr[i+1]-1]
    //   sources        : 0-based source-node indices (one per trunk centroid)
    //   labels         : label to assign to nodes reached via each source (e.g. TreeID)
    // Returns: IntegerVector of length n with the assigned label per node
    //          (NA_integer_ for unreachable nodes).
    // [[Rcpp::export]]
    IntegerVector shs_dijkstra_multi_cpp(IntegerVector adj_idx,
                                         NumericVector adj_w,
                                         IntegerVector adj_ptr,
                                         IntegerVector sources,
                                         IntegerVector labels) {
      const int n = adj_ptr.size() - 1;
      const int s = sources.size();
      const double INF = std::numeric_limits<double>::infinity();
      std::vector<double> dist(n, INF);
      std::vector<int> lab(n, NA_INTEGER);
      // Min-heap: (distance, node)
      typedef std::pair<double,int> P;
      std::priority_queue<P, std::vector<P>, std::greater<P> > pq;
      for (int i = 0; i < s; ++i) {
        int u = sources[i];
        if (u < 0 || u >= n) continue;
        if (0.0 < dist[u]) {
          dist[u] = 0.0;
          lab[u]  = labels[i];
          pq.push(std::make_pair(0.0, u));
        }
      }
      while (!pq.empty()) {
        P top = pq.top(); pq.pop();
        double d = top.first;
        int u = top.second;
        if (d > dist[u]) continue;       // stale
        int p0 = adj_ptr[u], p1 = adj_ptr[u+1];
        for (int p = p0; p < p1; ++p) {
          int v = adj_idx[p];
          double nd = d + adj_w[p];
          if (nd < dist[v]) {
            dist[v] = nd;
            lab[v]  = lab[u];
            pq.push(std::make_pair(nd, v));
          }
        }
      }
      IntegerVector out(n);
      for (int i = 0; i < n; ++i) out[i] = lab[i];
      return out;
    }
  ', env = .shs_dijkstra_compiled)
  .shs_dijkstra_compiled$fn <- .shs_dijkstra_compiled$shs_dijkstra_multi_cpp
  invisible(NULL)
}

segment_trees_shs <- function(las, bases,
                              trunk_zmax    = 4.0,
                              canopy_voxel  = 0.10,
                              knn_k         = 8L,
                              knn_max_dist  = 0.50) {
  if (!requireNamespace("RANN", quietly = TRUE))
    stop("seg_method='shs' requires the RANN package.")
  .shs_compile()
  pts <- as.matrix(las@data[, .(X, Y, Z)])
  base_xy <- as.matrix(bases[, c("X", "Y")])
  is_trunk <- pts[, 3] <= trunk_zmax
  tree_id <- rep(NA_integer_, nrow(pts))

  # Step 1: trunk slice -> nearest base XY.
  if (any(is_trunk)) {
    nn_t <- RANN::nn2(base_xy, pts[is_trunk, 1:2, drop = FALSE], k = 1L)
    tree_id[is_trunk] <- bases$TreeID[nn_t$nn.idx[, 1]]
  }
  message(sprintf("SHS step 1 (trunk slice): %d points labeled.", sum(is_trunk)))

  if (!any(!is_trunk)) {
    las@data$TreeID <- tree_id
    return(las)
  }

  # Step 2a: voxel-thin the canopy slice to keep the graph tractable.
  can_full <- pts[!is_trunk, , drop = FALSE]
  if (canopy_voxel > 0) {
    vox <- floor(can_full / canopy_voxel)
    key <- paste(vox[, 1], vox[, 2], vox[, 3], sep = ":")
    keep <- !duplicated(key)
    can <- can_full[keep, , drop = FALSE]
    message(sprintf("SHS step 2a (canopy thin %.2f m): %d -> %d nodes.",
                    canopy_voxel, nrow(can_full), nrow(can)))
  } else {
    can <- can_full
    message(sprintf("SHS step 2a (no thinning): %d canopy nodes.", nrow(can)))
  }

  # Append trunk centroids as graph nodes so they act as Dijkstra sources.
  # Each centroid is given Z = trunk_zmax (top of trunk slice) so it can
  # connect to nearby canopy points within knn_max_dist.
  base_pts <- cbind(base_xy, trunk_zmax)
  nodes <- rbind(base_pts, can)
  n_base <- nrow(base_pts)
  n_nodes <- nrow(nodes)

  # Step 2b: 3D KNN adjacency. Edges with d > knn_max_dist are dropped.
  message(sprintf("SHS step 2b (KNN k=%d, n=%d): building adjacency...",
                  knn_k + 1L, n_nodes))
  nn <- RANN::nn2(nodes, nodes, k = knn_k + 1L)   # +1 for self-match
  # Drop self (col 1) and over-distance edges. Build CSR.
  ni <- nn$nn.idx[, -1, drop = FALSE]            # n x k
  nd <- nn$nn.dists[, -1, drop = FALSE]          # n x k
  valid <- nd <= knn_max_dist
  deg <- rowSums(valid)
  adj_ptr <- as.integer(c(0L, cumsum(deg)))
  total_edges <- adj_ptr[length(adj_ptr)]
  adj_idx <- integer(total_edges)
  adj_w   <- numeric(total_edges)
  # Flatten valid edges row by row. Vectorized via t() on logical mask.
  idx_t <- t(ni); val_t <- t(valid); d_t <- t(nd)
  sel <- as.logical(val_t)
  adj_idx <- as.integer(idx_t[sel] - 1L)         # 0-based for C++
  adj_w   <- as.numeric(d_t[sel])
  message(sprintf("SHS step 2b: %d directed edges (mean degree %.1f, %d isolated nodes).",
                  total_edges, total_edges / n_nodes, sum(deg == 0L)))

  # Step 2c: multi-source Dijkstra from the n_base centroid nodes.
  sources <- seq_len(n_base) - 1L                # 0-based
  labels  <- as.integer(bases$TreeID)
  message(sprintf("SHS step 2c: Dijkstra from %d sources over %d nodes...",
                  n_base, n_nodes))
  t0 <- proc.time()[3]
  node_lab <- .shs_dijkstra_compiled$fn(
    adj_idx = adj_idx, adj_w = adj_w, adj_ptr = adj_ptr,
    sources = as.integer(sources), labels = labels
  )
  message(sprintf("SHS step 2c: Dijkstra done in %.1f s. %d / %d nodes reached.",
                  proc.time()[3] - t0, sum(!is.na(node_lab)), n_nodes))

  # Drop the trunk-centroid sentinel nodes; keep canopy node labels.
  can_lab <- node_lab[(n_base + 1L):n_nodes]

  # Step 2d: propagate labels back to the full (un-thinned) canopy slice.
  if (canopy_voxel > 0) {
    nn_back <- RANN::nn2(can, can_full, k = 1L)
    full_can_lab <- can_lab[nn_back$nn.idx[, 1]]
  } else {
    full_can_lab <- can_lab
  }
  tree_id[!is_trunk] <- full_can_lab

  # Any unreachable canopy points (graph isolated) fall back to nearest
  # base XY -- otherwise downstream forest_inventory chokes on NA TreeIDs.
  na_mask <- is.na(tree_id) & !is_trunk
  if (any(na_mask)) {
    nn_fb <- RANN::nn2(base_xy, pts[na_mask, 1:2, drop = FALSE], k = 1L)
    tree_id[na_mask] <- bases$TreeID[nn_fb$nn.idx[, 1]]
    message(sprintf("SHS fallback: %d unreachable canopy points -> nearest base XY.",
                    sum(na_mask)))
  }

  las@data$TreeID <- tree_id
  message(sprintf("seg_method='shs': %d trunk + %d canopy points; %d trees.",
                  sum(is_trunk), sum(!is_trunk), nrow(bases)))
  las
}

# ----------------------------------------------------------------------------
# TreeIso: native R/Rcpp port of the CloudCompare qTreeIso plugin
# (Xi & Hopkinson 2022, doi:10.3390/rs14236116). Three-stage cut-pursuit
# pipeline. Ignores `bases` and produces its own tree count.
# ----------------------------------------------------------------------------
segment_trees_treeisonet <- function(las, tls_params) {
  if (!requireNamespace("treeAIBoxR", quietly = TRUE))
    stop("seg_method='treeisonet' requires the treeAIBoxR package. ",
         "Install with devtools::install_local('R/modules/treeAIBoxR').")

  device <- if (identical(tls_params$treeisonet_device, "auto")) {
    if (torch::cuda_is_available()) "cuda" else "cpu"
  } else {
    tls_params$treeisonet_device
  }

  pts <- as.matrix(las@data[, c("X", "Y", "Z")])

  # Build voxel resolution override from params (NULL fields = use model default)
  vox_override <- list(
    xy = tls_params$treeisonet_vox_xy,
    z  = tls_params$treeisonet_vox_z
  )

  # ---- Optional TreeFiltering pre-filter ----------------------------------
  # When treeisonet_treefilter_model is set, run it first and restrict all
  # downstream steps to overstory points (class == 2). This mirrors the GUI
  # workflow where the treefilter scalar field gates StemCls and TreeLoc.
  tf_model_name <- tls_params$treeisonet_treefilter_model
  treefilter_idx <- NULL   # NULL = use all points
  if (!is.null(tf_model_name) && !is.na(tf_model_name) && nzchar(tf_model_name)) {
    tf_device <- if (identical(tls_params$treeisonet_treefilter_device, "auto")) {
      if (torch::cuda_is_available()) "cuda" else "cpu"
    } else {
      tls_params$treeisonet_treefilter_device
    }
    message("  [TreeFilter] Loading model: ", tf_model_name)
    tf_bundle <- treeAIBoxR::load_treeaibox_model(
      model_name = tf_model_name, device = tf_device)
    message("  [TreeFilter] Running classification...")
    tf_labels <- treeAIBoxR::classify_wood(
      xyz            = pts,
      model          = tf_bundle,
      if_bottom_only = isTRUE(tls_params$treeisonet_treefilter_bottom_only),
      verbose        = FALSE)
    treefilter_idx <- which(tf_labels > 1L)  # overstory points only
    message(sprintf("  [TreeFilter] Done: %d / %d points kept as overstory (%.1f%%).",
                    length(treefilter_idx), nrow(pts),
                    100 * length(treefilter_idx) / nrow(pts)))
    if (length(treefilter_idx) == 0L) {
      warning("treeisonet: TreeFiltering kept no overstory points.")
      return(las)
    }
    pts <- pts[treefilter_idx, , drop = FALSE]
  }

  # Auto-detect pipeline from model knobs:
  #   stemcls set  → TLS/UAV:  StemCls → TreeLoc → shortestpath3D
  #   treeoff set  → ALS:      TreeLoc → TreeOff → mergeshift
  use_als_path <- !is.na(tls_params$treeisonet_treeoff_model) &&
                  nzchar(tls_params$treeisonet_treeoff_model)
  scene_label  <- if (use_als_path) "ALS/reclamation" else "TLS/UAV"
  message(sprintf("seg_method='treeisonet' (%s): %d points, device=%s.",
                  scene_label, nrow(pts), device))
  t0 <- Sys.time()

  if (use_als_path) {
    # ========== ALS PIPELINE: TreeLoc → TreeOff → mergeshift ==============
    treeloc_name <- tls_params$treeisonet_treeloc_model
    if (is.na(treeloc_name) || !nzchar(treeloc_name))
      stop("treeisonet_treeloc_model is not set in tls_params.")

    message("  [1/3] Loading TreeLoc model: ", treeloc_name)
    treeloc_bundle <- treeAIBoxR::load_treeaibox_model(
      model_name = treeloc_name, device = device)

    message("  [1/3] Running TreeLoc detection...")
    base_locs <- treeAIBoxR::treeisonet_run_treeloc(
      xyz           = pts,
      model_bundle  = treeloc_bundle,
      cutoff_thresh = tls_params$treeisonet_cutoff_thresh,
      cut_grid_res  = tls_params$treeisonet_cut_grid_res,
      conf_thresh   = tls_params$treeisonet_conf_thresh,
      nms_thresh_xy = tls_params$treeisonet_nms_thresh_xy,
      base_radius_m = tls_params$treeisonet_base_radius_m,
      treeloc_max_gap = tls_params$treeisonet_treeloc_max_gap,
      treeloc_K     = tls_params$treeisonet_treeloc_K,
      vox_override  = vox_override,
      verbose       = TRUE)
    if (is.null(base_locs) || nrow(base_locs) == 0L) {
      warning("treeisonet: TreeLoc detected no bases.")
      return(las)
    }
    message(sprintf("  [1/3] TreeLoc done: %d bases.", nrow(base_locs)))

    treeoff_name <- tls_params$treeisonet_treeoff_model
    message("  [2/3] Loading TreeOff model: ", treeoff_name)
    treeoff_bundle <- treeAIBoxR::load_treeaibox_model(
      model_name = treeoff_name, device = device)

    message("  [2/3] Running TreeOff offset regression...")
    offsets <- treeAIBoxR::treeisonet_run_treeoff(
      xyz          = pts,
      treelocs     = base_locs,
      model_bundle = treeoff_bundle,
      vox_override = vox_override,
      verbose      = TRUE)

    message("  [3/3] Running mergeshift assignment...")
    tree_ids <- treeAIBoxR::treeisonet_mergeshift(
      xyz      = pts,
      offsets  = offsets,
      treelocs = base_locs)

  } else {
    # ========== TLS/UAV PIPELINE: StemCls → TreeLoc → shortestpath3D ======
    stemcls_name <- tls_params$treeisonet_stemcls_model
    if (is.na(stemcls_name) || !nzchar(stemcls_name))
      stop("treeisonet_stemcls_model is not set in tls_params.")

    message("  [1/3] Loading StemCls model: ", stemcls_name)
    stemcls_bundle <- treeAIBoxR::load_treeaibox_model(
      model_name = stemcls_name, device = device)
    message("  [1/3] Running StemCls classification...")
    stemcls <- treeAIBoxR::classify_wood(pts, stemcls_bundle, verbose = FALSE)
    n_stem <- sum(stemcls >= 2L)
    message(sprintf("  [1/3] StemCls done: %d stem points (%.1f%%).",
                    n_stem, 100 * n_stem / length(stemcls)))

    if (isTRUE(tls_params$treeisonet_stemcls_remove_small) && n_stem > 0L) {
      message("  [1/3] Removing small stem clusters (max_gap=",
              tls_params$treeisonet_stemcls_max_gap, " m, min_pts=",
              tls_params$treeisonet_stemcls_min_points, ")...")
      stemcls <- treeAIBoxR::treeisonet_remove_small_clusters(
        stemcls    = stemcls,
        xyz        = pts,
        max_gap_m  = tls_params$treeisonet_stemcls_max_gap,
        min_points = tls_params$treeisonet_stemcls_min_points)
      n_stem2 <- sum(stemcls >= 2L)
      message(sprintf("  [1/3] After cluster removal: %d stem points (%.1f%%).",
                      n_stem2, 100 * n_stem2 / length(stemcls)))
    }

    treeloc_name <- tls_params$treeisonet_treeloc_model
    if (is.na(treeloc_name) || !nzchar(treeloc_name))
      stop("treeisonet_treeloc_model is not set in tls_params.")
    message("  [2/3] Loading TreeLoc model: ", treeloc_name)
    treeloc_bundle <- treeAIBoxR::load_treeaibox_model(
      model_name = treeloc_name, device = device)
    message("  [2/3] Running TreeLoc detection...")
    base_locs <- treeAIBoxR::treeisonet_run_treeloc(
      xyz           = pts,
      model_bundle  = treeloc_bundle,
      cutoff_thresh = tls_params$treeisonet_cutoff_thresh,
      cut_grid_res  = tls_params$treeisonet_cut_grid_res,
      conf_thresh   = tls_params$treeisonet_conf_thresh,
      nms_thresh_xy = tls_params$treeisonet_nms_thresh_xy,
      base_radius_m = tls_params$treeisonet_base_radius_m,
      treeloc_max_gap = tls_params$treeisonet_treeloc_max_gap,
      treeloc_K     = tls_params$treeisonet_treeloc_K,
      vox_override  = vox_override,
      verbose       = TRUE)
    if (is.null(base_locs) || nrow(base_locs) == 0L) {
      warning("treeisonet: TreeLoc detected no bases — falling back to ",
              "nearest_base assignment.")
      return(las)
    }
    message(sprintf("  [2/3] TreeLoc done: %d bases detected.", nrow(base_locs)))

    message("  [3/3] Running shortestpath3D...")
    tree_ids <- treeAIBoxR::treeisonet_shortestpath3D(
      xyz               = pts,
      stemcls           = stemcls,
      base_locs         = base_locs,
      min_res           = tls_params$treeisonet_min_res,
      max_isolated_dist = tls_params$treeisonet_max_isolated_dist,
      k_graph           = tls_params$treeisonet_k_graph,
      k_node            = tls_params$treeisonet_k_node,
      verbose           = TRUE)
  }

  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  n_trees <- length(unique(tree_ids[tree_ids > 0L]))
  message(sprintf("  Done: %d trees on %d pts (%.1f s).",
                  n_trees, nrow(pts), elapsed))

  # Scatter tree_ids and stemcls back to full LAS.
  # When TreeFiltering was active, pts is a subset of the full cloud; we
  # zero-fill unvisited points (TreeID=0, StemCls=1 foliage).
  n_full <- nrow(las@data)
  full_tree_ids <- integer(n_full)       # 0 = unassigned
  full_stemcls  <- integer(n_full) + 1L  # 1 = foliage (safe default)

  if (!is.null(treefilter_idx)) {
    full_tree_ids[treefilter_idx] <- as.integer(tree_ids)
    if (exists("stemcls"))
      full_stemcls[treefilter_idx] <- as.integer(stemcls)
  } else {
    full_tree_ids <- as.integer(tree_ids)
    if (exists("stemcls"))
      full_stemcls <- as.integer(stemcls)
  }

  las@data$TreeID  <- full_tree_ids
  # StemCls: written for QSM use. 1=foliage/branch, 2=stem.
  # Also written for ALS path (stemcls not computed; stays 1 throughout).
  las@data$StemCls <- full_stemcls
  # TreeFilterLabel: 2L = overstory (passed to WoodCls / QSM), 1L = understory/ground.
  # NA when TreeFilter was not run (all points treated as overstory downstream).
  if (!is.null(treefilter_idx)) {
    tfl <- integer(n_full) + 1L
    tfl[treefilter_idx] <- 2L
    las@data$TreeFilterLabel <- tfl
  }
  las
}

segment_trees_treeiso <- function(las, bases) {
  if (!requireNamespace("treeisoR", quietly = TRUE)) {
    stop("seg_method='treeiso' requires the treeisoR package. ",
         "Install with devtools::install_local('R/modules/treeisoR').")
  }
  pts  <- as.matrix(las@data[, c("X", "Y", "Z")])
  message(sprintf("seg_method='treeiso': running 3-stage cut-pursuit on %d points...",
                  nrow(pts)))
  t0 <- Sys.time()
  ids <- treeisoR::treeiso_segment(
    pts,
    K1          = as.integer(tls_params$treeiso_K1),
    lambda1     = as.numeric(tls_params$treeiso_lambda1),
    dec_r1      = as.numeric(tls_params$treeiso_dec_r1),
    K2          = as.integer(tls_params$treeiso_K2),
    lambda2     = as.numeric(tls_params$treeiso_lambda2),
    max_gap     = as.numeric(tls_params$treeiso_max_gap),
    dec_r2      = as.numeric(tls_params$treeiso_dec_r2),
    K3          = as.integer(tls_params$treeiso_K3),
    rel_h_len_r = as.numeric(tls_params$treeiso_rel_h_len_r),
    vert_w      = as.numeric(tls_params$treeiso_vert_w),
    threads     = as.integer(tls_params$treeiso_threads),
    verbose     = isTRUE(tls_params$treeiso_verbose))
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  las@data$TreeID <- as.integer(ids)
  message(sprintf("seg_method='treeiso': %d trees on %d points (%.1f s; %d bases ignored).",
                  length(unique(ids[ids > 0])), nrow(pts), elapsed, nrow(bases)))
  las
}

method <- tolower(if (is.null(tls_params$seg_method)) "csp" else tls_params$seg_method)
if (!method %in% c("csp", "nearest_base", "shs", "treeiso", "treeisonet"))
  stop(sprintf("Unknown seg_method='%s'. Use 'csp', 'nearest_base', 'shs', 'treeiso', or 'treeisonet'.",
               method))
message(sprintf("Segmenting trees: method='%s', %d bases.", method, nrow(bases)))
las <- switch(method,
  csp           = csp_cost_segmentation(las, bases,
                    Voxel_size = tls_params$csp_voxel,
                    V_w        = tls_params$csp_v_w,
                    L_w        = tls_params$csp_l_w,
                    S_w        = tls_params$csp_s_w,
                    N_cores    = tls_params$csp_n_cores),
  nearest_base  = segment_trees_nearest_base(las, bases),
  shs           = segment_trees_shs(las, bases,
                    trunk_zmax   = tls_params$shs_trunk_zmax,
                    canopy_voxel = tls_params$shs_canopy_voxel,
                    knn_k        = tls_params$shs_knn_k,
                    knn_max_dist = tls_params$shs_knn_max_dist),
  treeiso       = segment_trees_treeiso(las, bases),
  treeisonet    = segment_trees_treeisonet(las, tls_params)
)


# 4. WoodCls  (wood / foliage classification)  +  understory StemCls
# ============================================================================
# WoodCls separates wood (branches + trunk, label>=2) from foliage (label=1)
# on the FULL cloud (overstory + understory). TreeFilterLabel is used afterward
# in forest_component_labels() to assign the correct Xi class:
#   overstory wood  -> class 2 (branch) or class 1 (stem, via StemCls)
#   understory wood -> class 6 (below-canopy branch) or class 5 (sapling stem)
# The QSM step (Section 9) still uses only overstory WoodLabel==2 points.
# Skip WoodCls by setting tls_params$woodcls_model = NA_character_.

if (!is.na(tls_params$woodcls_model) && nzchar(tls_params$woodcls_model)) {
  wc_device <- if (identical(tls_params$woodcls_device, "auto")) {
    if (torch::cuda_is_available()) "cuda" else "cpu"
  } else tls_params$woodcls_device

  message(sprintf("[4] WoodCls: loading model '%s' on %s...",
                  tls_params$woodcls_model, wc_device))
  wc_bundle  <- treeAIBoxR::load_treeaibox_model(
    model_name = tls_params$woodcls_model, device = wc_device)

  # Run on the full cloud so understory wood is also labelled (Xi classes 5/6).
  wc_pts <- as.matrix(las@data[, c("X", "Y", "Z")])
  message(sprintf("[4] WoodCls: classifying %d points (full cloud)...", nrow(wc_pts)))
  wc_labels <- treeAIBoxR::classify_wood(wc_pts, wc_bundle, verbose = FALSE)
  las@data$WoodLabel <- as.integer(wc_labels)

  n_wood <- sum(wc_labels >= 2L)
  message(sprintf("[4] WoodCls done: %d / %d points classified as wood (%.1f%%).",
                  n_wood, nrow(wc_pts), 100 * n_wood / nrow(wc_pts)))
  rm(wc_pts, wc_labels); invisible(gc())   # keep wc_bundle alive for downed log

  # -- Downed log detection (Xi class 4) ------------------------------------
  # Step 1: WoodCls with if_bottom_only=TRUE (2D XY voxelisation) on all
  #   near-ground non-ground points. In 2D mode the sliding blocks cover full
  #   vertical columns in XY, so horizontal logs span multiple XY blocks just
  #   as they do in the Xi 2023 implementation.
  # Step 2: PCA tilt > downed_tilt_min_deg (60 deg per Xi 2023) on the wood
  #   candidates to reject leaning stems whose bases are near the ground.
  # Written to las@data$DownedLog (logical); forest_component_labels() reads it.
  if (isTRUE(tls_params$downed_log_enable)) {
    dl_z_max  <- as.numeric(tls_params$downed_z_max)
    dl_tilt   <- as.numeric(tls_params$downed_tilt_min_deg)
    # PCA neighbourhood radius: 20x the WoodCls voxel resolution gives ~0.5 m
    # at the default 2.5 cm model, scaling automatically with coarser models.
    dl_res    <- wc_bundle$cfg$voxel_resolution_in_meter[[1L]]
    dl_radius <- 20.0 * dl_res
    dl_min_n  <- 8L    # minimum wood pts for stable SVD

    # All non-ground points below the height ceiling
    dl_cand <- which(las@data$Classification != 2L &
                     las@data$Z <= dl_z_max)
    las@data$DownedLog <- FALSE

    if (length(dl_cand) > 0L) {
      dl_pts <- as.matrix(las@data[dl_cand, c("X", "Y", "Z")])
      message(sprintf("[4] Downed log: WoodCls 2D pass on %d near-ground points...",
                      length(dl_cand)))
      # if_bottom_only=TRUE: 2D XY sliding blocks (Xi 2023 implementation)
      dl_wood <- treeAIBoxR::classify_wood(dl_pts, wc_bundle,
                                           if_bottom_only = TRUE,
                                           verbose = FALSE)
      wood_idx <- which(dl_wood >= 2L)   # wood candidates from 2D pass
      message(sprintf("[4] Downed log: %d / %d near-ground points classified as wood.",
                      length(wood_idx), length(dl_cand)))

      if (length(wood_idx) >= dl_min_n &&
          requireNamespace("RANN", quietly = TRUE)) {
        message(sprintf("[4] Downed log: PCA tilt filter on %d wood candidates...",
                        length(wood_idx)))
        xyz_all  <- as.matrix(las@data[, c("X", "Y", "Z")])
        xyz_wood <- dl_pts[wood_idx, , drop = FALSE]
        # Wood-only flag over full cloud for neighbourhood restriction
        is_wood  <- las@data$WoodLabel >= 2L & las@data$Classification != 2L
        nn_idx <- RANN::nn2(xyz_all, xyz_wood,
                            k = min(100L, nrow(xyz_all)),
                            searchtype = "radius",
                            radius = dl_radius)$nn.idx
        is_downed <- logical(length(wood_idx))
        for (i in seq_along(wood_idx)) {
          nbrs <- nn_idx[i, ]
          nbrs <- nbrs[nbrs > 0L]
          nbrs <- nbrs[is_wood[nbrs]]   # wood-only neighbourhood
          if (length(nbrs) < dl_min_n) next
          nb_xyz <- xyz_all[nbrs, , drop = FALSE]
          nb_xyz <- sweep(nb_xyz, 2L, colMeans(nb_xyz))
          sv <- svd(nb_xyz, nu = 0L, nv = 1L)$v[, 1L]
          cos_a    <- abs(sv[3L]) / sqrt(sum(sv^2))
          tilt_deg <- acos(pmin(1.0, cos_a)) * 180.0 / pi
          is_downed[i] <- tilt_deg >= dl_tilt
        }
        # Map wood_idx -> original dl_cand -> full-cloud indices
        las@data$DownedLog[dl_cand[wood_idx[is_downed]]] <- TRUE
        message(sprintf("[4] Downed log: %d / %d wood candidates flagged as downed (%.1f%%).",
                        sum(is_downed), length(wood_idx), 100 * mean(is_downed)))
        rm(xyz_all, xyz_wood, nn_idx, is_downed)
      } else if (length(wood_idx) > 0L) {
        # Too few wood points for PCA; accept the 2D model output directly
        las@data$DownedLog[dl_cand[wood_idx]] <- TRUE
        message("[4] Downed log: too few wood points for PCA tilt; using 2D model output.")
      }
      rm(dl_pts, dl_wood, wood_idx)
    } else {
      message("[4] Downed log: no near-ground non-ground points found.")
    }
    rm(dl_cand); invisible(gc())
  } else {
    message("[4] Downed log detection skipped (downed_log_enable = FALSE).")
  }
  rm(wc_bundle); invisible(gc())   # done with WoodCls model

  # -- Understory StemCls pass ----------------------------------------------
  # Run the same StemCls model on understory points (TreeFilterLabel==1) so
  # sapling stems (Xi class 5) can be distinguished from below-canopy branches
  # (Xi class 6). Overstory StemCls was already written to las@data$StemCls by
  # segment_trees_treeisonet(); only understory rows are updated here.
  us_stemcls_model <- tls_params$treeisonet_stemcls_model
  has_treefilter   <- "TreeFilterLabel" %in% names(las@data)
  if (!is.na(us_stemcls_model) && nzchar(us_stemcls_model) && has_treefilter) {
    us_idx <- which(las@data$TreeFilterLabel == 1L &
                    las@data$Classification != 2L)   # non-ground understory
    if (length(us_idx) > 0L) {
      message(sprintf("[4] Understory StemCls: classifying %d understory points...",
                      length(us_idx)))
      sc_device <- if (identical(tls_params$treeisonet_device, "auto")) {
        if (torch::cuda_is_available()) "cuda" else "cpu"
      } else tls_params$treeisonet_device
      sc_bundle <- treeAIBoxR::load_treeaibox_model(
        model_name = us_stemcls_model, device = sc_device)
      us_pts    <- as.matrix(las@data[us_idx, c("X", "Y", "Z")])
      us_labels <- treeAIBoxR::classify_wood(us_pts, sc_bundle, verbose = FALSE)
      # Only update understory rows; overstory StemCls stays intact.
      las@data$StemCls[us_idx] <- as.integer(us_labels)
      n_us_stem <- sum(us_labels >= 2L)
      message(sprintf("[4] Understory StemCls done: %d / %d understory stem points (%.1f%%).",
                      n_us_stem, length(us_idx),
                      100 * n_us_stem / length(us_idx)))
      rm(us_pts, sc_bundle, us_labels, us_idx); invisible(gc())
    } else {
      message("[4] Understory StemCls skipped: no non-ground understory points found.")
    }
  } else if (!has_treefilter) {
    message("[4] Understory StemCls skipped: TreeFilter was not run (no TreeFilterLabel column).")
  }
} else {
  message("[4] WoodCls skipped (tls_params$woodcls_model is NA).")
  # Write a default WoodLabel of 2 (all points treated as wood) so Section 9
  # can still run if manually triggered with woodcls_model = NA.
  if (!"WoodLabel" %in% names(las@data))
    las@data$WoodLabel <- 2L
}


# ============================================================================
# 5. Per-tree QC PNGs
# ============================================================================
# Four-panel side views per TreeID written to <out_dir>/<tree_qc_subdir>/.
# Top row: RGB XZ / RGB YZ (or component colours if no RGB).
# Bottom row: forest_component_labels() colouring (8-class palette).
# Not all classes will appear per tree -- legend is dynamic.
# Runs in batch (builds qc_las locally; does not require Section 7 rgl).
if (isTRUE(tls_params$tree_qc_enable)) {
  qc_dir <- file.path(out_dir, tls_params$tree_qc_subdir)
  if (dir.exists(qc_dir))
    invisible(file.remove(list.files(qc_dir, pattern = "\\.png$", full.names = TRUE)))
  dir.create(qc_dir, showWarnings = FALSE, recursive = TRUE)
  has_inv <- exists("inv") && is.data.frame(inv)
  # Always rebuild from the current las so WoodLabel / StemCls / DownedLog
  # are up-to-date.  A stale las_seg from a previous interactive Section 7
  # run would be missing those columns.
  qc_las <- lidR::filter_poi(las, !is.na(TreeID) & TreeID > 0L)
  data.table::setDT(qc_las@data)
  qc_ids <- sort(unique(qc_las@data$TreeID))
  qc_ids <- qc_ids[!is.na(qc_ids)]
  if (!is.na(tls_params$tree_qc_top_n) &&
      tls_params$tree_qc_top_n < length(qc_ids)) {
    if (has_inv) {
      ord <- order(-inv$DBH[match(qc_ids, inv$TreeID)], na.last = TRUE)
      qc_ids <- qc_ids[ord][seq_len(tls_params$tree_qc_top_n)]
    } else {
      qc_ids <- qc_ids[seq_len(tls_params$tree_qc_top_n)]
    }
  }
  hw <- tls_params$tree_qc_xy_halfwidth
  has_rgb <- isTRUE(tls_params$tree_qc_use_rgb) &&
             all(c("R", "G", "B") %in% names(qc_las@data)) &&
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
    if (nrow(pts) < tls_params$tree_qc_min_points) next
    inv_row <- if (has_inv) inv[inv$TreeID == id, ] else data.frame()
    dbh <- if (nrow(inv_row) && "DBH"    %in% names(inv_row)) inv_row$DBH[1]    else NA_real_
    ht  <- if (nrow(inv_row) && "Height" %in% names(inv_row)) inv_row$Height[1] else NA_real_
    fid <- if (nrow(inv_row) && "f_id" %in% names(inv_row)) inv_row$f_id[1] else NA
    cx     <- median(pts$X); cy <- median(pts$Y)
    ylim_z <- range(pts$Z, na.rm = TRUE)   # full tree height, not clipped by asp
    if (has_rgb) {
      r8 <- pmin(pmax(pts$R / rgb_div, 0), 1)
      g8 <- pmin(pmax(pts$G / rgb_div, 0), 1)
      b8 <- pmin(pmax(pts$B / rgb_div, 0), 1)
      rgb_col <- grDevices::rgb(r8, g8, b8)
      fc_lbl <- forest_component_labels(pts)
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
           main = sprintf("RGB YZ  DBH=%.1fcm HT=%.1fm pts=%d  field=%s",
                          ifelse(is.na(dbh), NA, dbh * 100),
                          ht, nrow(pts),
                          ifelse(is.na(fid), "none", as.character(fid))),
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
      fc_lbl <- forest_component_labels(pts)
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
           main = sprintf("TreeID %d  XZ  DBH=%.1fcm HT=%.1fm pts=%d  field=%s",
                          id, ifelse(is.na(dbh), NA, dbh * 100), ht, nrow(pts),
                          ifelse(is.na(fid), "none", as.character(fid))),
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
# 6. FOREST INVENTORY
# ============================================================================
# Fits circles in thin slices and splines DBH/X/Y vs Z. Returns one row per
# TreeID with X, Y, DBH, Height, ConvexHullArea, quality_flag.

# 6.1 Patch CspStandSegmentation::forest_inventory ---------------------------
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


# 6.2 Run inventory + DBH/Height filter --------------------------------------
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
# 7. INTERACTIVE PLOTTING
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

  # Single tree -- pick by the label you see in the plot. Default to the
  # tallest matched TreeID; override `tid` to inspect a specific tree.
  tid <- if (exists("inv") && nrow(inv) > 0) inv$TreeID[which.max(inv$Height)] else NA_integer_
  tree <- las_seg;  tree@data <- las_seg@data[TreeID == tid]
  if (nrow(tree@data) == 0L) {
    warning("Single-tree view: TreeID ", tid, " has no segmented points; skipping.")
  } else {
    y <- lidR::plot(tree, color = "RGB", size = 2, axis = FALSE)
    draw_axes(y, tree@data,
              ceiling(inv$Height[inv$TreeID == tid] / 5) * 5, step_xy = 1)
  }

  # 5b. Full-plot QC -------------------------------------------------------
  # rgl view of the entire segmented cloud. For mask-based QC overlays:
  #   plot_cloud_qc(las, mask = drop_mask)
  plot_cloud_qc(las, color_by = "XiLabel",
                title = "Point decomposition (Xi 2023 classes)")
}


# 7c. Note: per-tree QC PNGs are in Section 5 (after WoodCls/Section 4).

# ---- wood/foliage classifier helpers (used by Section 5) ----------------
# Local-PCA linearity classifier. Per point, take its k nearest neighbours,
# build the 3x3 covariance, eigenvalue-decompose, compute
# linearity = (l1 - l2) / l1. Above the threshold => wood.
classify_wf_pca <- function(xyz, k = 10L, lin_thr = 0.85,
                            max_pts = 60000L) {
  n <- nrow(xyz)
  if (n < (k + 1L)) return(rep(TRUE, n))   # too few points: assume foliage
  # Subsample for speed; classify on the subsample, vote labels back to
  # full set by nearest-neighbour lookup.
  if (n > max_pts) {
    samp <- sort(sample.int(n, max_pts))
    sub  <- xyz[samp, , drop = FALSE]
  } else {
    samp <- seq_len(n)
    sub  <- xyz
  }
  nn  <- RANN::nn2(sub, sub, k = k + 1L, treetype = "kd")$nn.idx
  lin <- numeric(nrow(sub))
  for (i in seq_len(nrow(sub))) {
    nbrs <- sub[nn[i, ], , drop = FALSE]
    cv   <- cov(nbrs)
    ev   <- sort(eigen(cv, symmetric = TRUE, only.values = TRUE)$values,
                 decreasing = TRUE)
    lin[i] <- if (ev[1] <= 0) 0 else (ev[1] - ev[2]) / ev[1]
  }
  is_wood_sub <- lin > lin_thr
  if (n == nrow(sub)) return(!is_wood_sub)   # is_foliage = !is_wood
  # Vote labels back to the full point set via 1-NN
  back <- RANN::nn2(sub, xyz, k = 1L)$nn.idx[, 1L]
  !is_wood_sub[back]
}

# Lazy loader for the TreeAIBox backend. Caches the model in the parent
# (tree_qc) scope across the per-tree loop.
load_treeaibox_lazy <- function(tls_params) {
  if (!requireNamespace("treeAIBoxR", quietly = TRUE))
    stop("treeAIBoxR is not installed. Install from R/modules/treeAIBoxR ",
         "or set tls_params$tree_qc_wf_method = 'pca'.", call. = FALSE)
  dev <- tls_params$tree_qc_treeaibox_device
  if (identical(dev, "auto"))
    dev <- if (torch::cuda_is_available()) "cuda" else "cpu"
  # Prefer model_name (triggers auto-download) over explicit paths
  mn <- tls_params$tree_qc_treeaibox_model
  if (!is.null(mn) && !is.na(mn) && nzchar(mn)) {
    return(treeAIBoxR::load_treeaibox_model(
      model_name = mn,
      device     = dev,
      strict     = FALSE))
  }
  # Fall back to explicit weights/config paths (backwards compatible)
  wp <- tls_params$tree_qc_treeaibox_weights
  cp <- tls_params$tree_qc_treeaibox_config
  if (is.null(wp) || is.na(wp) || !nzchar(wp))
    stop("Set tree_qc_treeaibox_model (preferred) or tree_qc_treeaibox_weights.",
         call. = FALSE)
  if (is.null(cp) || is.na(cp) || !nzchar(cp))
    stop("tree_qc_treeaibox_config is not set in tls_params.", call. = FALSE)
  treeAIBoxR::load_treeaibox_model(
    weights_path = wp,
    config_path  = cp,
    device       = dev,
    strict       = FALSE)
}

# Returns logical vector of length nrow(pts): TRUE = foliage.
classify_wf <- function(pts, tls_params, treeaibox_model = NULL) {
  method <- tls_params$tree_qc_wf_method
  xyz <- as.matrix(pts[, c("X", "Y", "Z"), with = FALSE])
  if (identical(method, "pca")) {
    return(classify_wf_pca(xyz,
                           k       = tls_params$tree_qc_wf_k,
                           lin_thr = tls_params$tree_qc_wf_linearity,
                           max_pts = tls_params$tree_qc_wf_max_pts))
  }
  if (identical(method, "treeaibox")) {
    if (is.null(treeaibox_model))
      stop("classify_wf: treeaibox_model must be supplied for this method.",
           call. = FALSE)
    labs <- treeAIBoxR::classify_wood(xyz, treeaibox_model)
    # Convention from componentFilter: 1 = foliage; >1 = wood (binary or
    # multiclass with branch/stem). is_foliage == (label == 1).
    return(labs == 1L)
  }
  stop("Unknown tree_qc_wf_method: ", method, call. = FALSE)
}


# ============================================================================
# 8. FIELD-DATA VALIDATION
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
    stop("Section 8 requires the readxl package.")

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
    stop("Section 8: no field rows after plot filter.")

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
        inv$azi <- (inv$azi - bias) %% 360
        inv$X   <- inv$distance_m * sin(inv$azi * pi / 180)
        inv$Y   <- inv$distance_m * cos(inv$azi * pi / 180)
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
  message("Section 8 skipped (tls_params$field_xlsx_path is NA or missing).")
}


# ============================================================================
# 9. QSM — TreeAIBox applyQSM pipeline
# ============================================================================
# Builds a per-tree Quantitative Structure Model (QSM) using cut-pursuit
# over-segmentation + Dijkstra skeleton + algebraic circle-fit radii.
# Matches the TreeAIBox GUI "Apply QSM" workflow (applyQSM.py).
#
# Prerequisites:
#   - Section 3 must have run (las@data$TreeID and las@data$StemCls present)
#   - Section 4 WoodCls should have run (las@data$WoodLabel present);
#     if WoodLabel is absent all points are treated as wood.
#   - treeisoR package must be rebuilt with cut_pursuit_segment() exported.
#     (Run: install_treeisoR.R or reinstall treeisoR when no R session is open)
#
# Output:
#   - Per-tree RDS files: <out_dir>/qsm/tree_<id>.rds  → apply_qsm() result
#   - Per-tree cylinder CSV: <out_dir>/qsm/tree_<id>_cylinders.csv
#   - Combined cylinder CSV: <out_dir>/qsm/all_trees_cylinders.csv

local({
  # Guard: TreeID column is mandatory
  if (!"TreeID" %in% names(las@data)) {
    message("[8] QSM skipped: las@data$TreeID not present ",
            "(run Section 3 first).")
    return(invisible(NULL))
  }

  # Check treeisoR has cut_pursuit_segment exported
  if (!exists("cut_pursuit_segment", envir = asNamespace("treeisoR"),
              inherits = FALSE)) {
    message("[8] QSM skipped: treeisoR::cut_pursuit_segment() not available. ",
            "Rebuild treeisoR (close all R sessions, then R CMD INSTALL ",
            "R/modules/treeisoR) and re-source this script.")
    return(invisible(NULL))
  }

  # Which trees to process
  all_tree_ids <- sort(unique(las@data$TreeID[las@data$TreeID > 0L]))
  qsm_ids <- if (all(is.na(tls_params$qsm_tree_ids))) {
    all_tree_ids
  } else {
    intersect(as.integer(tls_params$qsm_tree_ids), all_tree_ids)
  }
  if (length(qsm_ids) == 0L) {
    message("[8] QSM: no valid tree IDs to process.")
    return(invisible(NULL))
  }

  qsm_dir <- if (!is.na(tls_params$qsm_out_dir)) {
    tls_params$qsm_out_dir
  } else {
    file.path(out_dir, "qsm")
  }
  dir.create(qsm_dir, showWarnings = FALSE, recursive = TRUE)

  has_woodlabel <- "WoodLabel" %in% names(las@data)
  has_stemcls   <- "StemCls"   %in% names(las@data)

  all_cyl <- vector("list", length(qsm_ids))
  message(sprintf("[8] QSM: processing %d tree(s)...", length(qsm_ids)))

  for (ii in seq_along(qsm_ids)) {
    tid <- qsm_ids[ii]
    message(sprintf("  [8] Tree %d (%d/%d)...", tid, ii, length(qsm_ids)))

    tree_mask  <- las@data$TreeID == tid
    wood_mask  <- if (has_woodlabel) las@data$WoodLabel >= 2L else rep(TRUE, nrow(las@data))
    pts_mask   <- tree_mask & wood_mask

    if (sum(pts_mask) < 20L) {
      message(sprintf("    Skipping tree %d: only %d wood points.", tid, sum(pts_mask)))
      next
    }

    # Build (n, 4) matrix [X, Y, Z, stemcls]
    d <- las@data[pts_mask, ]
    stemcls_col <- if (has_stemcls) as.integer(d$StemCls) else rep(1L, nrow(d))
    pts_qsm <- cbind(as.numeric(d$X), as.numeric(d$Y), as.numeric(d$Z),
                     stemcls_col)

    # Centre coordinates for numerical stability (add back at output)
    xyz_offset <- colMeans(pts_qsm[, 1:3, drop = FALSE])
    pts_qsm[, 1L] <- pts_qsm[, 1L] - xyz_offset[1L]
    pts_qsm[, 2L] <- pts_qsm[, 2L] - xyz_offset[2L]
    pts_qsm[, 3L] <- pts_qsm[, 3L] - xyz_offset[3L]

    qsm_res <- tryCatch(
      treeAIBoxR::apply_qsm(
        pts                  = pts_qsm,
        k_neighbors          = tls_params$qsm_k_neighbors,
        max_graph_distance   = tls_params$qsm_max_graph_distance,
        max_conn_dist        = tls_params$qsm_max_conn_dist,
        occlusion_cutoff     = tls_params$qsm_occlusion_cutoff,
        min_pts_clean        = tls_params$qsm_min_pts_clean,
        K_stem               = tls_params$qsm_K_stem,
        reg_stem             = tls_params$qsm_reg_stem,
        K_branch             = tls_params$qsm_K_branch,
        reg_branch           = tls_params$qsm_reg_branch,
        min_radius_m         = tls_params$qsm_min_radius_m,
        threads              = tls_params$qsm_threads,
        verbose              = TRUE),
      error = function(e) {
        message(sprintf("    ERROR tree %d: %s", tid, conditionMessage(e)))
        NULL
      })

    if (is.null(qsm_res)) next

    # Save raw QSM result (for aRchi or further analysis)
    saveRDS(qsm_res, file.path(qsm_dir, sprintf("tree_%d.rds", tid)))

    # Convert to cylinder table and save CSV
    cyl <- treeAIBoxR::qsm_to_cylinder_table(qsm_res, tree_id = tid,
                                              xyz_offset = xyz_offset)
    if (nrow(cyl) > 0L) {
      write.csv(cyl, file.path(qsm_dir, sprintf("tree_%d_cylinders.csv", tid)),
                row.names = FALSE)
      all_cyl[[ii]] <- cyl
      n_cyl <- nrow(cyl)
      # DBH estimate: cylinder closest to 1.3 m HAG (Z = min(Z) + 1.3)
      z_base <- min(cyl$startZ, na.rm = TRUE)
      dbh_row <- which.min(abs(cyl$startZ - (z_base + 1.3)))
      dbh_est <- 2 * cyl$radius[dbh_row]
      vol_est <- sum(pi * cyl$radius^2 * cyl$length, na.rm = TRUE)
      message(sprintf("    Tree %d: %d cylinders | DBH ~%.3f m | vol ~%.4f m3",
                      tid, n_cyl, dbh_est, vol_est))
    }
  }

  # Combine and save all trees
  all_cyl <- Filter(Negate(is.null), all_cyl)
  if (length(all_cyl) > 0L) {
    all_cyl_df <- do.call(rbind, all_cyl)
    write.csv(all_cyl_df,
              file.path(qsm_dir, "all_trees_cylinders.csv"),
              row.names = FALSE)
    message(sprintf("[8] QSM complete: %d trees, %d cylinders total. Output: %s",
                    length(all_cyl), nrow(all_cyl_df), qsm_dir))
  }
})


# ============================================================================
# ============================================================================

