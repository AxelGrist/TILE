# Load Required packages ----
library(lidR)
library(future)
library(terra)
library(sf)
library(viridis)
library(rgl)
library(ggplot2)
library(whitebox)
library(httr)
library(jsonlite)
library(parallel)

# CONFIGURATION ----

## Project Paths
main_dir        <- "D:/alpine_treemapping/TILE/outputs"    # Main output directory
whitebox_path   <- "C:/Program Files/WhiteboxTools_win_amd64/WBT/whitebox_tools.exe"

## Area of Interest
aoi_shapefile   <- NULL  # Path to AOI shapefile (e.g., "D:/data/study_area.shp"), or NULL to skip clipping

## LiDAR Download Settings
download_config <- list(
  enabled        = FALSE,                    # Set TRUE to run download
  use_aoi        = TRUE,                     # Use AOI shapefile for spatial query (recommended)
  most_recent    = TRUE,                     # Download most recent data available for each tile
  min_year       = NULL,                     # Minimum year filter (e.g., 2020), or NULL for any
  # Manual tile selection (only used if use_aoi = FALSE)
  map_tile       = "093",                    # NTS map tile (e.g., "092", "093")
  letter         = "h",                      # NTS letter
  year           = 2020,                     # Specific year (only if most_recent = FALSE)
  tile_numbers   = c("040"),                 # Tile numbers or "all"
  subdivisions   = "all",                    # Subdivisions or "all"
  # Output settings
  download_dir   = "D:/lidar_downloads",     # Where to save downloaded files
  parallel       = TRUE,                     # Use parallel downloads
  max_workers    = 8                         # Max parallel download workers
)

## LiDAR Source Folders
source_folders <- c(
  "V:/FOR_RNI_RNI_Projects/Research/Data_Lidar/Lidar/Omineca NE Data Transfered to GeoBC/PG_Final_Products/2020/2020_RE218210005_Robson_TSA/LiDAR",
  "V:/FOR_RNI_RNI_Projects/Research/Data_Lidar/Lidar/Omineca NE Data Transfered to GeoBC/PG_Final_Products/2020/2020_RE218210006_Robson_TSA/LiDAR"
)

## Target Files (Set to NULL to use all .laz files in source_folders)
target_filenames <- c(
  "bc_093h040_1_4_3_xyes_8_bcalb_2020.laz",
  "bc_093h040_1_4_4_xyes_8_bcalb_2020.laz",
  "bc_093h040_2_3_3_xyes_8_bcalb_2020.laz",
  "bc_093h040_1_4_1_xyes_8_bcalb_2020.laz",
  "bc_093h040_1_4_2_xyes_8_bcalb_2020.laz",
  "bc_093h040_2_3_1_xyes_8_bcalb_2020.laz"
)

## Processing Parameters
processing_config <- list(
  crs            = 3005,       # Coordinate reference system (BC Albers)
  chunk_size     = 100,        # Chunk size in meters
  chunk_buffer   = 10,         # Buffer around chunks (meters)
  resolution     = 0.15,       # Raster resolution (meters)
  hmin           = 2,          # Minimum tree height (meters)
  hmax           = 30,         # Maximum tree height (meters)
  min_points     = 5,          # Minimum LiDAR points per tree
  uniqueness     = "bitmerge", # Tree ID uniqueness method (bitmerge recommended)
  bbox_size      = 100,        # Sample pipeline subset size (meters)
  hillshade_angle     = 45,    # Hillshade sun elevation angle (degrees)
  hillshade_direction = 315    # Hillshade sun azimuth direction (degrees, 315 = NW)
)

## Parallel Processing
parallel_config <- list(
  workers        = 1,          # Number of parallel workers for catalog processing
  lidr_threads   = 8           # Threads for lidR operations
)

# INITIALIZATION ----

# Initialize WhiteBox Tools
if (file.exists(whitebox_path)) {
  wbt_init(whitebox_path)
  cat("✓ WhiteBox Tools initialized\n")
} else {
  warning("WhiteBox Tools not found at: ", whitebox_path)
}

# Validate and create output directories
if (!dir.exists(main_dir)) {
  dir.create(main_dir, recursive = TRUE)
  cat("✓ Created output directory:", main_dir, "\n")
}

# Load AOI if specified
aoi <- NULL
if (!is.null(aoi_shapefile)) {
  if (file.exists(aoi_shapefile)) {
    aoi <- st_read(aoi_shapefile, quiet = TRUE)
    aoi <- st_transform(aoi, processing_config$crs)
    cat("✓ AOI loaded:", aoi_shapefile, "\n")
    cat("  Extent:", paste(round(st_bbox(aoi), 1), collapse = ", "), "\n")
  } else {
    warning("AOI shapefile not found: ", aoi_shapefile)
  }
}

# Check download directory for existing files
if (download_config$enabled) {
  if (dir.exists(download_config$download_dir)) {
    existing_files <- list.files(download_config$download_dir, pattern = "\\.laz$", full.names = FALSE)
    if (length(existing_files) > 0) {
      cat("ℹ Found", length(existing_files), "existing .laz files in download directory\n")
    }
  }
}

# Validate source folders exist
valid_folders <- source_folders[dir.exists(source_folders)]
if (length(valid_folders) == 0) {
  warning("No valid source folders found. Check paths in source_folders.")
} else if (length(valid_folders) < length(source_folders)) {
  cat("⚠ Some source folders not found. Using", length(valid_folders), "of", length(source_folders), "folders.\n")
}


# Download LiDAR from LidarBC using AOI shapefile (spatial query)
download_lidar_by_aoi <- function(aoi_sf, download_dir, most_recent = TRUE, min_year = NULL,
                                   parallel_download = TRUE, max_workers = 8) {
  
  base_api <- "https://services1.arcgis.com/xeMpV7tU1t4KD3Ei/arcgis/rest/services/LidarBC_Open_LIDAR/FeatureServer/4/query"
  
  if (!dir.exists(download_dir)) dir.create(download_dir, recursive = TRUE)
  
  # Transform AOI to Web Mercator (EPSG:3857) for ArcGIS query
  aoi_3857 <- st_transform(aoi_sf, 3857)
  
  # Get bounding box and build envelope geometry
  bbox <- st_bbox(aoi_3857)
  geometry_json <- sprintf(
    '{"xmin":%f,"ymin":%f,"xmax":%f,"ymax":%f,"spatialReference":{"wkid":3857}}',
    bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
  )
  
  # Build WHERE clause
  where_clause <- "1=1"  # Get all tiles in spatial extent
  if (!is.null(min_year)) {
    where_clause <- paste0("year >= ", min_year)
  }
  
  cat("Querying LidarBC for tiles intersecting AOI...\n")
  cat("  Bounding box (Web Mercator):", paste(round(bbox, 1), collapse = ", "), "\n")
  
  # Query with pagination (API returns max 2000 records)
  all_attrs <- NULL
  offset <- 0
  repeat {
    resp <- GET(base_api, query = list(
      where = where_clause,
      geometry = geometry_json,
      geometryType = "esriGeometryEnvelope",
      spatialRel = "esriSpatialRelIntersects",
      outFields = "filename,year,s3Url,maptile,path",
      returnGeometry = "false",
      resultOffset = offset,
      resultRecordCount = 2000,
      f = "json"
    ))
    
    if (status_code(resp) != 200) stop("Failed to query ArcGIS API. Status: ", status_code(resp))
    
    data <- fromJSON(content(resp, "text", encoding = "UTF-8"))
    
    if (length(data$features) == 0) break
    
    all_attrs <- rbind(all_attrs, data$features$attributes)
    
    if (is.null(data$exceededTransferLimit) || !data$exceededTransferLimit) break
    offset <- offset + 2000
    cat("  Fetched", nrow(all_attrs), "records, continuing...\n")
  }
  
  if (is.null(all_attrs) || nrow(all_attrs) == 0) {
    cat("No LiDAR tiles found for the specified AOI.\n")
    return(NULL)
  }
  
  cat("✓ Found", nrow(all_attrs), "total LiDAR tiles in AOI\n")
  
  # Get most recent data for each tile location if requested
  if (most_recent) {
    cat("  Filtering to most recent data per tile...\n")
    # Extract base tile (without year info) - e.g., "093h040_1_4_3"
    all_attrs$base_tile <- all_attrs$maptile
    
    # Keep only most recent year for each base tile
    all_attrs <- all_attrs[order(all_attrs$base_tile, -all_attrs$year), ]
    all_attrs <- all_attrs[!duplicated(all_attrs$base_tile), ]
    
    cat("✓ Filtered to", nrow(all_attrs), "most recent tiles\n")
    cat("  Year range:", min(all_attrs$year), "-", max(all_attrs$year), "\n")
  }
  
  # Build download URLs - use s3Url field (available for all years)
  urls <- all_attrs$s3Url
  
  # Handle any missing s3Url by constructing from path
  missing_url <- is.na(urls) | urls == "" | is.null(urls)
  if (any(missing_url)) {
    # Construct URL from path field
    paths_fixed <- gsub("\\\\", "/", all_attrs$path[missing_url])
    urls[missing_url] <- paste0("https://nrs.objectstore.gov.bc.ca/gdwuts", paths_fixed)
  }
  
  filenames <- all_attrs$filename
  
  # Check for existing files
  existing <- file.exists(file.path(download_dir, filenames))
  if (any(existing)) {
    cat("ℹ Skipping", sum(existing), "already downloaded files\n")
    urls <- urls[!existing]
    filenames <- filenames[!existing]
  }
  
  if (length(urls) == 0) {
    cat("✓ All files already downloaded!\n")
    return(download_dir)
  }
  
  cat("Total files to download:", length(urls), "\n")
  
  # Estimate download size (rough: ~50MB per tile average)
  est_size_gb <- length(urls) * 50 / 1024
  cat(sprintf("  Estimated download size: ~%.1f GB\n", est_size_gb))
  
  # Confirmation if many files
  if (length(urls) > 100) {
    cat("WARNING: You are about to download", length(urls), "files.\n")
    cat("Type 'yes' to continue: ")
    response <- tolower(readline())
    if (response != "yes") {
      cat("Download cancelled.\n")
      return(NULL)
    }
  }
  
  # Download logic
  download_file <- function(i) {
    destfile <- file.path(download_dir, filenames[i])
    tryCatch({
      resp <- httr::GET(urls[i], httr::write_disk(destfile, overwrite = TRUE), 
                        httr::timeout(300))
      if (httr::status_code(resp) == 200) {
        return(list(success = TRUE, file = filenames[i]))
      } else {
        return(list(success = FALSE, file = filenames[i], status = httr::status_code(resp)))
      }
    }, error = function(e) {
      return(list(success = FALSE, file = filenames[i], error = e$message))
    })
  }
  
  if (parallel_download && length(urls) > 1) {
    cat("Starting parallel download with", min(max_workers, length(urls)), "workers...\n")
    cl <- makeCluster(min(max_workers, detectCores(), length(urls)))
    clusterExport(cl, varlist = c("urls", "filenames", "download_dir"), envir = environment())
    clusterEvalQ(cl, library(httr))
    
    results <- parLapply(cl, seq_along(urls), download_file)
    stopCluster(cl)
  } else {
    cat("Starting sequential download...\n")
    results <- lapply(seq_along(urls), function(i) {
      cat("\r  Downloading", i, "of", length(urls), ":", filenames[i], "    ")
      download_file(i)
    })
    cat("\n")
  }
  
  # Report results
  successes <- sum(sapply(results, function(x) x$success))
  failures <- length(results) - successes
  cat("✓ Download complete:", successes, "succeeded,", failures, "failed\n")
  
  if (failures > 0) {
    failed_files <- sapply(results[!sapply(results, function(x) x$success)], function(x) x$file)
    cat("  Failed files:", paste(head(failed_files, 5), collapse = ", "))
    if (length(failed_files) > 5) cat("... and", length(failed_files) - 5, "more")
    cat("\n")
  }
  
  return(download_dir)
}

# Download lidar files from the LidarBC portal (manual tile selection)
download_lidar_bc_manual <- function(map_tile, letter, year, tile_numbers = "all", subdivisions = "all",
                                     download_dir = "D:/test", parallel_download = TRUE, max_workers = 8) {
  
  # ArcGIS REST API endpoint for LiDAR Point Cloud Index
  base_api <- "https://services1.arcgis.com/xeMpV7tU1t4KD3Ei/arcgis/rest/services/LidarBC_Open_LIDAR/FeatureServer/4/query"
  
  if (!dir.exists(download_dir)) dir.create(download_dir, recursive = TRUE)
  
  # Handle "all" for subdivisions
  if (length(subdivisions) == 1 && subdivisions == "all") {
    grid <- expand.grid(a = 1:4, b = 1:4, c = 1:4)
    subdivisions <- apply(grid, 1, function(x) paste0(x[1], "_", x[2], "_", x[3]))
  }
  
  # Handle "all" for tile_numbers
  if (length(tile_numbers) == 1 && tile_numbers == "all") {
    tile_numbers <- sprintf("%03d", 0:999)
  } else {
    tile_numbers <- sprintf("%03d", as.numeric(tile_numbers))
  }
  
  # Build WHERE clause for ArcGIS query
  if (length(tile_numbers) == 1 && tile_numbers == "all") {
    where_clause <- paste0("year=", year, " AND maptile LIKE '", map_tile, tolower(letter), "%'")
  } else {
    patterns <- paste0("maptile LIKE '", map_tile, tolower(letter), tile_numbers, "%'")
    where_clause <- paste0("year=", year, " AND (", paste(patterns, collapse = " OR "), ")")
  }
  
  cat("Querying ArcGIS service...\n")
  resp <- GET(base_api, query = list(
    where = where_clause,
    outFields = "filename,year,s3Url,maptile,path",
    f = "json",
    returnGeometry = "false"
  ))
  
  if (status_code(resp) != 200) stop("Failed to query ArcGIS API.")
  
  data <- fromJSON(content(resp, "text", encoding = "UTF-8"))
  
  if (length(data$features) == 0) {
    cat("No features found for the specified criteria.\n")
    return(NULL)
  }
  
  # Extract attributes
  attrs <- data$features$attributes
  
  # Parse maptile into components
  attrs$tile_number <- sub(".*([0-9]{3}).*", "\\1", attrs$maptile)
  attrs$subdivision <- sub(".*_([0-9]_[0-9]_[0-9])$", "\\1", attrs$maptile)
  
  # Filter by subdivisions
  matched <- attrs[attrs$subdivision %in% subdivisions, ]
  
  if (nrow(matched) == 0) {
    cat("No matching files found.\n")
    return(NULL)
  }
  
  # Use s3Url field (works for all years)
  urls <- matched$s3Url
  
  # Handle any missing s3Url by constructing from path
  missing_url <- is.na(urls) | urls == "" | is.null(urls)
  if (any(missing_url)) {
    paths_fixed <- gsub("\\\\", "/", matched$path[missing_url])
    urls[missing_url] <- paste0("https://nrs.objectstore.gov.bc.ca/gdwuts", paths_fixed)
  }
  
  filenames <- matched$filename
  
  # Skip existing files
  existing <- file.exists(file.path(download_dir, filenames))
  if (any(existing)) {
    cat("ℹ Skipping", sum(existing), "already downloaded files\n")
    urls <- urls[!existing]
    filenames <- filenames[!existing]
  }
  
  if (length(urls) == 0) {
    cat("✓ All files already downloaded!\n")
    return(download_dir)
  }
  
  cat("Total files to download:", length(urls), "\n")
  
  # Confirmation if >100 files
  if (length(urls) > 100) {
    cat("WARNING: You are about to download", length(urls), "files.\n")
    cat("Type 'yes' to continue: ")
    response <- tolower(readline())
    if (response != "yes") {
      cat("Download cancelled.\n")
      return(NULL)
    }
  }
  
  # Download logic
  if (parallel_download && length(urls) > 1) {
    cat("Starting parallel download with", min(max_workers, length(urls)), "workers...\n")
    cl <- makeCluster(min(max_workers, detectCores(), length(urls)))
    clusterExport(cl, varlist = c("urls", "filenames", "download_dir"), envir = environment())
    clusterEvalQ(cl, library(httr))
    parLapply(cl, seq_along(urls), function(i) {
      destfile <- file.path(download_dir, filenames[i])
      resp <- httr::GET(urls[i], httr::write_disk(destfile, overwrite = TRUE), httr::timeout(300))
      if (httr::status_code(resp) == 200) {
        cat("Downloaded:", filenames[i], "\n")
      } else {
        cat("Failed:", filenames[i], "Status:", httr::status_code(resp), "\n")
      }
    })
    stopCluster(cl)
  } else {
    for (i in seq_along(urls)) {
      cat("\r  Downloading", i, "of", length(urls), ":", filenames[i], "    ")
      destfile <- file.path(download_dir, filenames[i])
      resp <- GET(urls[i], write_disk(destfile, overwrite = TRUE), timeout(300))
      if (status_code(resp) != 200) {
        cat("\n  Failed:", filenames[i], "Status:", status_code(resp), "\n")
      }
    }
    cat("\n")
  }
  
  return(download_dir)
}

# Run download if enabled
if (download_config$enabled) {
  cat("\n--- Running LiDAR Download ---\n")
  
  if (download_config$use_aoi && !is.null(aoi)) {
    # AOI-based spatial query (recommended)
    download_lidar_by_aoi(
      aoi_sf = aoi,
      download_dir = download_config$download_dir,
      most_recent = download_config$most_recent,
      min_year = download_config$min_year,
      parallel_download = download_config$parallel,
      max_workers = download_config$max_workers
    )
  } else if (!download_config$use_aoi) {
    # Manual tile selection
    download_lidar_bc_manual(
      map_tile = download_config$map_tile,
      letter = download_config$letter,
      year = download_config$year,
      tile_numbers = download_config$tile_numbers,
      subdivisions = download_config$subdivisions,
      download_dir = download_config$download_dir,
      parallel_download = download_config$parallel,
      max_workers = download_config$max_workers
    )
  } else {
    warning("Download enabled with use_aoi=TRUE but no AOI shapefile specified. Set aoi_shapefile path.")
  }
  
  # Add download directory to source folders
  source_folders <- c(source_folders, download_config$download_dir)
}

# Search folders for LAS/LAZ files
all_files <- unlist(lapply(source_folders[dir.exists(source_folders)], 
                           list.files, pattern = "\\.(laz|las)$", full.names = TRUE))

# Filter to target files if specified, otherwise use all
if (!is.null(target_filenames) && length(target_filenames) > 0) {
  matched_files <- all_files[basename(all_files) %in% target_filenames]
  cat("✓ Found", length(matched_files), "of", length(target_filenames), "target files\n")
} else {
  matched_files <- all_files
  cat("✓ Found", length(matched_files), "LAS/LAZ files\n")
}

# Validate we have files to process
if (length(matched_files) == 0) {
  stop("No LAS/LAZ files found. Check source_folders and target_filenames configuration.")
}

# Clip files to AOI if specified
if (!is.null(aoi)) {
  cat("ℹ AOI specified - files will be clipped during processing\n")
}

# Create LAS catalog
ctg <- catalog(matched_files)
las_check(ctg)

# Apply projection
projection(ctg) <- st_crs(processing_config$crs)$proj4string

# Set catalog options from config
opt_laz_compression(ctg) <- TRUE
opt_chunk_size(ctg) <- processing_config$chunk_size
opt_chunk_buffer(ctg) <- processing_config$chunk_buffer
opt_select(ctg) <- "*"
opt_filter(ctg) <- ""
opt_progress(ctg) <- TRUE

# Set parallel processing
plan(multisession, workers = parallel_config$workers)
set_lidr_threads(parallel_config$lidr_threads)
options(process_chunk.debug = TRUE)

cat("✓ Catalog created with", length(matched_files), "files\n")
cat("  Chunk size:", processing_config$chunk_size, "m | Buffer:", processing_config$chunk_buffer, "m\n")
cat("  Workers:", parallel_config$workers, "| Threads:", parallel_config$lidr_threads, "\n")


# Initialize Sample Pipeline
sample_pipeline <- function(las_file, main_dir = main_dir, 
                        bbox_size = processing_config$bbox_size, 
                        hmin = processing_config$hmin, 
                        hmax = processing_config$hmax, 
                        res = processing_config$resolution, 
                        min_points = processing_config$min_points,
                        uniqueness = processing_config$uniqueness,
                        crs = processing_config$crs,
                        hillshade_angle = processing_config$hillshade_angle,
                        hillshade_direction = processing_config$hillshade_direction) {
  
  # 1. Set output directories
  dirs <- list(
    gpkg = file.path(main_dir, "outputs/sample/trees"),
    dsm  = file.path(main_dir, "outputs/sample/dsm"),
    dtm  = file.path(main_dir, "outputs/sample/dtm"),
    chm  = file.path(main_dir, "outputs/sample/chm"),
    slope = file.path(main_dir, "outputs/sample/slope"),
    aspect = file.path(main_dir, "outputs/sample/aspect")
  )
  lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE)
  
  # 2. Read LAS file
  las_full <- readLAS(las_file)
  if (is.empty(las_full)) stop("LAS file is empty!")
  projection(las_full) <- st_crs(crs)$proj4string
  
  # 3. Pick random non-empty subset
  bb <- ext(las_full)
  attempts <- 0
  repeat {
    xmin <- runif(1, bb[1], bb[2] - bbox_size)
    ymin <- runif(1, bb[3], bb[4] - bbox_size)
    xmax <- xmin + bbox_size
    ymax <- ymin + bbox_size
    las_small <- clip_rectangle(las_full, xmin, ymin, xmax, ymax)
    if (!is.empty(las_small)) break
    if (++attempts > 20) stop("Could not find a non-empty subset.")
  }
  message(sprintf("Using subset: X [%f, %f], Y [%f, %f], Points: %d",
                  xmin, xmax, ymin, ymax, npoints(las_small)))
  
  prefix <- paste0(round(xmin), "_", round(ymin)) # Template name for output files
  
  # 4. Remove duplicate points
  las_small <- filter_duplicates(las_small)
  
  # 5. Generate DTM, slope, and aspect from ground classified points
  ground <- filter_poi(las_small, Classification == 2)
  if (!is.empty(ground)) {
    
    # 5a. Generate DTM
    dtm_tile <- rasterize_terrain(ground, algorithm = tin(), res = res)
    dtm_file <- file.path(dirs$dtm, paste0("DTM_", prefix, ".tif"))
    writeRaster(dtm_tile, dtm_file, overwrite = TRUE)
    
    # 5b. Slope and Aspect in radians (required for hillshade)
    slope_tile <- terrain(dtm_tile, v = "slope", unit = "radians", neighbors = 8)
    aspect_tile <- terrain(dtm_tile, v = "aspect", unit = "radians", neighbors = 8)
    
    # 5c. Hillshade
    hillshade <- shade(slope_tile, aspect_tile, angle = hillshade_angle, direction = hillshade_direction)
    
    # 5d. Convert slope and aspect tiles to degrees for plotting and extraction
    slope_tile <- slope_tile * 180 / pi
    aspect_tile <- aspect_tile * 180 / pi
    
    # 5e. Save slope and aspect in degrees
    writeRaster(slope_tile, file.path(dirs$slope, paste0("slope_", prefix, ".tif")), overwrite = TRUE)
    writeRaster(aspect_tile, file.path(dirs$aspect, paste0("aspect_", prefix, ".tif")), overwrite = TRUE)
    
  } else {
    dtm_tile <- slope_tile <- aspect_tile <- NULL
    dtm_file <- NA
  }
  
  # 6. Generate DSM
  dsm_tile <- rasterize_canopy(las_small, res = res, algorithm = p2r(na.fill = tin()), pkg = "terra")
  dsm_file <- file.path(dirs$dsm, paste0("DSM_", prefix, ".tif"))
  writeRaster(dsm_tile, dsm_file, overwrite = TRUE)
  
  # 7. Normalize point cloud
  las_small <- normalize_height(las_small, tin())
  
  # 8. Filter noise (normalized points below ground / above hmax)
  las_small <- filter_poi(las_small, Classification != 2)
  las_small <- filter_poi(las_small, Z >= 0 & Z <= hmax)
  
  # 9. Generate CHM
  chm <- rasterize_canopy(las_small, res = res,algorithm = pitfree(thresholds = c(0, hmax*0.5, hmax), subcircle = 0.2), pkg = "terra")
  if (all(is.na(values(chm)))) stop("CHM is empty. Cannot detect trees.")
  chm_file <- file.path(dirs$chm, paste0("CHM_", prefix, ".tif"))
  writeRaster(chm, chm_file, overwrite = TRUE)
  
  # 10a, Adaptive moving window function
  aw <- function(x) {
    y <- 2.6 * (-(exp(-0.08 * (x - hmin)) - 1)) + 3
    y[x < hmin] <- 3; y[x > hmax] <- 5
    y
  }
  
  # 10b. Treetop detection using adaptive window fnction
  treetops <- locate_trees(chm, lmf(ws = aw, shape = "circular"))
  
  # 11. Tree segmentation
  algo <- dalponte2016(chm, treetops, ID = "treeID")
  las_segmented <- segment_trees(las_small, algo, uniqueness = uniqueness)
  
  # 12a. Calculate crown metrics and save as points
  tree_metrics <- crown_metrics(las_segmented, func = .stdtreemetrics, geom = "point")
  
  # 12b. Remove all trees that have less than min_points lidar points
  tree_metrics <- tree_metrics[tree_metrics$npoints >= min_points, ]
  
  # 12c. Convert tree metrics points to sf
  tree_metrics_sf <- st_as_sf(tree_metrics)
  st_crs(tree_metrics_sf) <- st_crs(las_full)
  
  # 13. Extract elevation, slope and aspect at tree locations
  if (!is.null(dtm_tile)) {
    tree_metrics_sf$dtm_z      <- terra::extract(dtm_tile, vect(tree_metrics_sf))[,2]
    tree_metrics_sf$slope_deg  <- terra::extract(slope_tile, vect(tree_metrics_sf))[,2]
    tree_metrics_sf$aspect_deg <- terra::extract(aspect_tile, vect(tree_metrics_sf))[,2]
  } else {
    tree_metrics_sf$dtm_z <- tree_metrics_sf$slope_deg <- tree_metrics_sf$aspect_deg <- NA
  }
  
  # 14. Print tree metrics summary
  print(summary(tree_metrics_sf))
  
  # 15. Write trees GeoPackage 
  gpkg_file <- file.path(dirs$gpkg, paste0("trees_", prefix, ".gpkg"))
  st_write(tree_metrics_sf, gpkg_file, delete_dsn = TRUE, quiet = TRUE)
  
  # 16. Plot DTM + DSM + CHM + treetops ---
  if (!is.null(dtm_tile) & !is.null(dsm_tile) & !is.null(slope_tile) & !is.null(aspect_tile)) {
    par(mfrow = c(2, 3), mar = c(4, 4, 4, 6))
    tt_spat <- vect(treetops); crs(tt_spat) <- crs(dtm_tile)
    
    # DTM
    plot(dtm_tile, col = viridis(100), main = "DTM")
    plot(tt_spat, add = TRUE, col = "red", pch = 16, cex = 0.7)
    
    # DSM
    plot(dsm_tile, col = viridis(100), main = "DSM")
    plot(tt_spat, add = TRUE, col = "red", pch = 16, cex = 0.7)
    
    # CHM
    plot(chm, col = viridis(100), main = "CHM")
    plot(tt_spat, add = TRUE, col = "red", pch = 16, cex = 0.7)
    
    # Slope (°)
    plot(slope_tile, col =  viridis(100), main = "Slope (°)")
    plot(tt_spat, add = TRUE, col = "red", pch = 16, cex = 0.7)
    
    # Aspect (°)
    plot(aspect_tile, col = viridis(100), main = "Aspect (°)")
    plot(tt_spat, add = TRUE, col = "red", pch = 16, cex = 0.7)
    
    # Hillshade
    plot(hillshade, col =gray(0:30/30), main ='Hillshade')
    plot(tt_spat, add = TRUE, col = "red", pch = 16, cex = 0.7)
    
    par(mfrow = c(1, 1))
  }
  
  # 17. 3D plot of segmented LAS colored by treeID
  scene <- plot(las_segmented, bg = "black", size = 2, color = "treeID", bbox = TRUE, axis = TRUE)
  tt_sf <- st_as_sf(treetops, coords = c("X","Y"), crs = st_crs(las_segmented))
  add_treetops3d(scene, tt_sf, color = "red", size = 1)
  
  # 18. Pipeline Complete
  message(sprintf("Sample processing complete. Total trees detected: %d", nrow(treetops)))
  
}

# Catalog Pipeline
process_chunk <- function(las_chunk, main_dir, 
                          res = processing_config$resolution, 
                          hmin = processing_config$hmin, 
                          hmax = processing_config$hmax, 
                          min_points = processing_config$min_points, 
                          uniqueness = processing_config$uniqueness) {
  las <- if (inherits(las_chunk, "LAScluster")) readLAS(las_chunk) else las_chunk
  if (is.empty(las)) return(NULL)
  
  core_ext <- tryCatch({
    if (inherits(las_chunk, "LAScluster")) ext(las_chunk) else ext(las)
  }, error = function(e) ext(las))
  
  prefix <- paste0(round(xmin(core_ext)), "_", round(ymin(core_ext)))
  
  dirs <- list(
    dtm    = file.path(main_dir, "outputs/catalog/dtm"),
    dsm    = file.path(main_dir, "outputs/catalog/dsm"),
    chm    = file.path(main_dir, "outputs/catalog/chm"),
    slope  = file.path(main_dir, "outputs/catalog/slope"),
    aspect = file.path(main_dir, "outputs/catalog/aspect"),
    trees  = file.path(main_dir, "outputs/catalog/trees")
  )
  lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE)
  
  las <- filter_duplicates(las)
  
  dtm_tile <- slope_tile <- aspect_tile <- NULL
  if ("Classification" %in% names(las@data)) {
    ground <- filter_poi(las, Classification == 2)
    if (!is.empty(ground)) {
      dtm_full <- rasterize_terrain(ground, algorithm = tin(), res = res, pkg = "terra")
      slope_full  <- terrain(dtm_full, v = "slope",  unit = "degrees")
      aspect_full <- terrain(dtm_full, v = "aspect", unit = "degrees")
      
      dtm_tile    <- crop(dtm_full, core_ext)
      slope_tile  <- crop(slope_full, core_ext)
      aspect_tile <- crop(aspect_full, core_ext)
      
      writeRaster(dtm_tile,    file.path(dirs$dtm,    paste0("dtm_", prefix, ".tif")), overwrite = TRUE)
      writeRaster(slope_tile,  file.path(dirs$slope,  paste0("slope_",  prefix, ".tif")), overwrite = TRUE)
      writeRaster(aspect_tile, file.path(dirs$aspect, paste0("aspect_", prefix, ".tif")), overwrite = TRUE)
    }
  }
  
  dsm_full <- rasterize_canopy(las, res = res, algorithm = p2r(na.fill = tin()), pkg = "terra")
  dsm_tile <- crop(dsm_full, core_ext)
  writeRaster(dsm_tile, file.path(dirs$dsm, paste0("dsm_", prefix, ".tif")), overwrite = TRUE)
  
  las_norm <- tryCatch(normalize_height(las, tin()), error = function(e) NULL)
  chm_tile <- NULL
  las_ng <- NULL
  
  if (!is.null(las_norm)) {
    if ("Classification" %in% names(las_norm@data)) {
      las_ng <- filter_poi(las_norm, Classification != 2)
    } else {
      las_ng <- las_norm
    }
    if (!is.null(las_ng) && !is.empty(las_ng)) {
      las_ng <- filter_poi(las_ng, Z >= 0 & Z <= hmax)
    }
    
    if (!is.null(las_ng) && !is.empty(las_ng)) {
      chm_full <- rasterize_canopy(
        las_ng, res = res,
        algorithm = pitfree(thresholds = c(0, hmax * 0.5, hmax), subcircle = 0.2),
        pkg = "terra"
      )
      chm_tile <- crop(chm_full, core_ext)
    }
  }
  
  if (is.null(chm_tile) && !is.null(dtm_tile)) {
    chm_tile <- tryCatch({
      r <- dsm_tile - dtm_tile
      r[r < 0] <- NA
      r
    }, error = function(e) NULL)
  }
  
  if (!is.null(chm_tile)) {
    writeRaster(chm_tile, file.path(dirs$chm, paste0("chm_", prefix, ".tif")), overwrite = TRUE)
  }
  
  can_try_trees <- FALSE
  if (!is.null(chm_tile) && !is.null(las_ng) && !is.empty(las_ng)) {
    chm_max <- tryCatch(terra::global(chm_tile, "max", na.rm = TRUE)[1,1], error = function(e) NA_real_)
    if (is.finite(chm_max) && !is.na(chm_max) && chm_max >= hmin) can_try_trees <- TRUE
  }
  
  if (!can_try_trees) return(NULL)
  
  aw <- function(x) {
    y <- 2.6 * (-(exp(-0.08 * (x - hmin)) - 1)) + 3
    y[x < hmin] <- 3
    y[x > hmax] <- 5
    y
  }
  treetops <- tryCatch(locate_trees(chm_tile, lmf(ws = aw, shape = "circular")),
                       error = function(e) NULL)
  if (is.null(treetops) || nrow(treetops) == 0) return(NULL)
  
  algo <- dalponte2016(chm_tile, treetops, ID = "treeID")
  las_segmented <- tryCatch(segment_trees(las_ng, algo, uniqueness = uniqueness),
                            error = function(e) NULL)
  if (is.null(las_segmented) || is.empty(las_segmented)) return(NULL)
  
  las_core <- tryCatch(clip_roi(las_segmented, core_ext), error = function(e) las_segmented)
  if (!("treeID" %in% names(las_core@data))) return(NULL)
  las_core <- filter_poi(las_core, !is.na(treeID))
  if (is.empty(las_core)) return(NULL)
  
  tree_metrics <- tryCatch(
    crown_metrics(las_core, func = .stdtreemetrics, geom = "point"),
    error = function(e) NULL
  )
  if (is.null(tree_metrics) || nrow(tree_metrics) == 0) return(NULL)
  if ("npoints" %in% names(tree_metrics)) {
    tree_metrics <- tree_metrics[tree_metrics$npoints >= min_points, ]
  }
  if (nrow(tree_metrics) == 0) return(NULL)
  
  tree_sf <- if (inherits(tree_metrics, "sf")) tree_metrics else st_as_sf(tree_metrics)
  st_crs(tree_sf) <- tryCatch(sf::st_crs(projection(las)), error = function(e) sf::st_crs(3005))
  
  if (!is.null(dtm_tile) && nrow(tree_sf) > 0) {
    coords <- vect(tree_sf)
    ex_dtm    <- tryCatch(terra::extract(dtm_tile, coords),    error = function(e) NULL)
    ex_slope  <- tryCatch(terra::extract(slope_tile, coords),  error = function(e) NULL)
    ex_aspect <- tryCatch(terra::extract(aspect_tile, coords), error = function(e) NULL)
    
    tree_sf$dtm_z      <- if (!is.null(ex_dtm)    && ncol(ex_dtm)    >= 2) ex_dtm[[2]]    else rep(NA_real_, nrow(tree_sf))
    tree_sf$slope_deg  <- if (!is.null(ex_slope)  && ncol(ex_slope)  >= 2) ex_slope[[2]]  else rep(NA_real_, nrow(tree_sf))
    tree_sf$aspect_deg <- if (!is.null(ex_aspect) && ncol(ex_aspect) >= 2) ex_aspect[[2]] else rep(NA_real_, nrow(tree_sf))
  } else {
    tree_sf$dtm_z <- tree_sf$slope_deg <- tree_sf$aspect_deg <- NA_real_
  }
  
  gpkg_file <- file.path(dirs$trees, paste0("trees_", prefix, ".gpkg"))
  st_write(tree_sf, gpkg_file, delete_dsn = TRUE, quiet = TRUE)
  
  return(NULL)
}

# RUN SAMPLE PIPELINE----
test_file <- matched_files[1] # First matching las file
result <- sample_pipeline(test_file)
gc()



# RUN CATALOG PIPELINE ----
catalog_apply(ctg, process_chunk, 
             main_dir = main_dir, 
             res = processing_config$resolution,
             hmin = processing_config$hmin,
             hmax = processing_config$hmax, 
             min_points = processing_config$min_points,
             uniqueness = processing_config$uniqueness)
# CATALOG TROUBLESHOOTING ----
# Uncomment any code that may be useful:
# opt_restart(ctg) <- 1220  # Use the failed chunk number printed in console as a checkpoint to restart pipeline
# MOSAIC TILES----
mosaic_tiles <- function(main_dir) {
  mosaic_dir <- file.path(main_dir, "outputs/mosaic")
  dir.create(mosaic_dir, showWarnings = FALSE, recursive = TRUE)
  products <- c("dtm", "dsm", "chm", "slope", "aspect")
  for (prod in products) {
    in_dir <- file.path(main_dir, "outputs/catalog", prod)
    out_file <- file.path(mosaic_dir, paste0(prod, "_mosaic.tif"))
    files <- list.files(in_dir, pattern = "\\.tif$", full.names = TRUE)
    if (length(files) == 0) {
      message("No tiles found for ", prod)
      next
    }
    message("Mosaicking ", length(files), " tiles for ", prod, "...")
    r <- vrt(files)
    writeRaster(r, out_file, overwrite = TRUE,
                gdal = c("COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"))
    message("✓ Saved: ", out_file)
  }
  # Merge tree GPKGs
  trees_dir <- file.path(main_dir, "outputs/catalog/trees")
  trees_out <- file.path(mosaic_dir, "trees_merged.gpkg")
  tfiles <- list.files(trees_dir, pattern = "\\.gpkg$", full.names = TRUE)
  if (length(tfiles) > 0) {
    message("Merging ", length(tfiles), " tree GPKGs...")
    sflist <- lapply(tfiles, function(f) st_read(f, quiet = TRUE))
    
    all_cols <- Reduce(union, lapply(sflist, names))
    sflist2 <- lapply(sflist, function(x) {
      miss <- setdiff(all_cols, names(x))
      for (m in miss) x[[m]] <- NA
      x[, all_cols]
    })
    trees_merged <- do.call(rbind, sflist2)
    st_write(trees_merged, trees_out, delete_dsn = TRUE, quiet = TRUE)
    message("✓ Trees merged: ", trees_out)
  } else {
    message("No tree GPKGs found in: ", trees_dir)
  }
}
# MOSAIC PROCESSING----

# This section processes and generates additional rasters that require the entire area as context

# Tree density function
tree_density <- function(main_dir, window_size = 100) {
  # Define input and output paths
  trees_file   <- file.path(main_dir, "outputs/mosaic/trees_merged.gpkg")
  raster_file  <- file.path(main_dir, "outputs/mosaic/dtm_mosaic.tif")
  output_file  <- file.path(main_dir, "outputs/mosaic/tree_density_per_hectare.tif")
  
  # Validate inputs exist
  if (!file.exists(trees_file)) stop("Trees file not found: ", trees_file)
  if (!file.exists(raster_file)) stop("Raster file not found: ", raster_file)
  
  message("Loading tree points...")
  trees <- st_read(trees_file, quiet = TRUE)
  message("  ", nrow(trees), " trees loaded")
  
  message("Loading reference raster...")
  r <- rast(raster_file)
  
  message("Rasterizing tree points...")
  tree_raster <- rasterize(vect(trees), r, field = 1, background = 0)
  
  message("Calculating tree density (", window_size, "m x ", window_size, "m window)...")
  # Create square window (window_size x window_size meters = 1 hectare if 100m)
  window_matrix <- focalMat(r, d = window_size, type = "circle")
  
  # Apply focal count (sum of tree points in window)
  density_raster <- focal(tree_raster, w = window_matrix, fun = "sum", na.rm = TRUE)
  
  message("Writing output...")
  writeRaster(density_raster, output_file, overwrite = TRUE,
              gdal = c("COMPRESS=LZW", "TILED=YES"))
  message("✓ Tree density raster saved to: ", output_file)
  
  return(invisible(density_raster))
}

# Run tree density (uncomment to execute - can be slow for large areas)
# tree_density(main_dir = main_dir, window_size = 100)

# Time in daylight
# This tool is very CPU intensive. If resources are limited consider lowering max distance and start/end days/times.
# Load the DSM (or DEM if trees/vegetation shadows are not a concern)
dsm <- rast(file.path(main_dir, "outputs/mosaic/dsm_mosaic.tif"))
# Get extent and calculate center in projected coordinates
ext <- ext(dsm)
center_x <- (ext[1] + ext[2]) / 2
center_y <- (ext[3] + ext[4]) / 2
# Create a point in EPSG:3005
center_point <- vect(cbind(center_x, center_y), crs = crs(dsm))
# Transform to geographic coordinates (EPSG:4326)
center_point_geo <- project(center_point, "EPSG:4326")
# Extract numeric latitude and longitude
long <- geom(center_point_geo)[, "x"]
lat <- geom(center_point_geo)[, "y"]
# Time In Daylight Main Function
wbt_time_in_daylight(
  dem = file.path(main_dir, "outputs/mosaic/dsm_mosaic.tif"), # Use DSM if looking at effects of shadows on vegetation
  output = file.path(main_dir, "outputs/mosaic/time_in_daylight_mosaic"),
  lat = lat,
  long = long,
  az_fraction = 10,
  max_dist = 100,
  utc_offset = "-08:00",
  start_day = 1,
  end_day = 365,
  start_time = "00:00:00",
  end_time = "23:59:59",
)

# STATISTICS ----

# --- Load once ---
gpkg <- file.path(main_dir, "outputs/mosaic/trees_merged.gpkg")
trees <- st_read(gpkg, quiet = TRUE)
df <- st_drop_geometry(trees)

# --- Column names (edit if needed) ---
height_col <- "Z"
area_col   <- "convhull_area"
elev_col   <- "dtm_z"
slope_col  <- "slope_deg"
aspect_col <- "aspect_deg"
npts_col   <- "npoints"

# --- Derived: northness from aspect (handles circular aspect) ---
# northness = cos(aspect in radians): 1=N, 0=E/W, -1=S
df$northness <- cos((df[[aspect_col]] %% 360) * pi / 180)

# --- Reusable plotting helper (2D binned heatmap) ---
plot_bin2d <- function(df, x_col, y_col, x_lab, y_lab, title,
                       bins = 400, binwidth = NULL) {
  d <- df[is.finite(df[[x_col]]) & is.finite(df[[y_col]]), ]
  ggplot(d, aes_string(x = x_col, y = y_col)) +
    { if (is.null(binwidth)) geom_bin2d(bins = bins) else geom_bin2d(binwidth = binwidth) } +
    scale_fill_viridis_c(trans = "log10", name = "Count") +
    labs(x = x_lab, y = y_lab, title = title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))
}


# 1) Height vs Slope
p_height_slope <- plot_bin2d(
  df, slope_col, height_col,
  x_lab = "Slope (degrees)", y_lab = "Tree Height (m)",
  title = "Tree Height vs Slope"
)

# 2) Height vs Northness (aspect effect)
p_height_north <- plot_bin2d(
  df, "northness", height_col,
  x_lab = "Northness (cos aspect)", y_lab = "Tree Height (m)",
  title = "Tree Height vs Northness"
)

# 3) Crown area vs Height
p_area_height <- plot_bin2d(
  df, height_col, area_col,
  x_lab = "Tree Height (m)", y_lab = "Convex Hull Area (m²)",
  title = "Crown Area vs Tree Height"
)

# 4) Crown area vs Elevation
p_area_elev <- plot_bin2d(
  df, elev_col, area_col,
  x_lab = "Elevation (m)", y_lab = "Convex Hull Area (m²)",
  title = "Crown Area vs Elevation"
)

# 5) Crown area vs Slope
p_area_slope <- plot_bin2d(
  df, slope_col, area_col,
  x_lab = "Slope (degrees)", y_lab = "Convex Hull Area (m²)",
  title = "Crown Area vs Slope"
)

# 6) npoints vs Height
p_npts_height <- plot_bin2d(
  df, height_col, npts_col,
  x_lab = "Tree Height (m)", y_lab = "Point Count",
  title = "Point Count vs Tree Height"
)

# 7) npoints vs Crown area
p_npts_area <- plot_bin2d(
  df, area_col, npts_col,
  x_lab = "Convex Hull Area (m²)", y_lab = "Point Count",
  title = "Point Count vs Crown Area"
)

# 8) npoints vs Elevation
p_npts_elev <- plot_bin2d(
  df, elev_col, npts_col,
  x_lab = "Elevation (m)", y_lab = "Point Count",
  title = "Point Count vs Elevation"
)

# 9) Height vs elevation
p_height_elev <-  plot_bin2d(
  df, elev_col, height_col,
  x_lab = "Elevation (m)", y_lab = "Tree Height (m)",
  title = "Tree Height vs Elevation"
)

# Print any you want to inspect:
p_height_slope
p_height_north
p_area_height
p_area_elev
p_area_slope
p_npts_height
p_npts_area
p_npts_elev
p_height_elev



daylight <- rast(file.path(main_dir, "outputs/mosaic/time_in_daylight_mosaic.tif"))
plot(daylight)
