# TILE: Tree Inventory and Landscape Evaluation

A LiDAR processing tool for individual tree detection and terrain analysis in British Columbia.

## Overview

TILE processes airborne LiDAR data to generate:
- **Terrain Products**: Digital Terrain Model (DTM), Digital Surface Model (DSM), hillshade, slope, aspect
- **Canopy Products**: Canopy Height Model (CHM)
- **Tree Inventory**: Individual tree detection with location, height, and crown metrics

## Features

- **LiDAR Download**: Automated download from BC's Open LiDAR portal by AOI or map tile
- **Terrain Processing**: DTM/DSM generation using WhiteBox Tools
- **Tree Detection**: Individual tree segmentation using lidR
- **Parallel Processing**: Multi-threaded processing for large datasets
- **Catalog-based**: Efficient processing of multiple LiDAR tiles

## Requirements

- **R** (>= 4.0)
- **R Packages**: lidR, terra, sf, whitebox, future, parallel
- **WhiteBox Tools**: [Download here](https://www.whiteboxgeo.com/download-whiteboxtools/)

### Install R Packages

```r
install.packages(c("lidR", "terra", "sf", "whitebox", "future", 
                   "parallel", "viridis", "ggplot2", "httr", "jsonlite"))
```

## Project Structure

```
TILE/
├── TILE.Rproj          # RStudio project file
├── README.md           # This file
├── R/
│   └── TILE.R          # Main processing script
└── outputs/
    ├── catalog/        # Terrain rasters (DTM, DSM, CHM, slope, aspect)
    ├── mosaic/         # Merged outputs
    ├── sample/         # Sample/test outputs
    ├── temp/           # Temporary processing files
    └── pdf/            # PDF reports
```

## Usage

1. Open `TILE.Rproj` in RStudio
2. Edit configuration in `R/TILE.R`:
   - Set `main_dir` to your output directory
   - Set `source_folders` to your LiDAR data locations
   - Configure `processing_config` parameters
3. Run the script

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resolution` | Output raster resolution (meters) | 0.15 |
| `hmin` | Minimum tree height (meters) | 2 |
| `hmax` | Maximum tree height (meters) | 30 |
| `chunk_size` | Processing chunk size (meters) | 100 |
| `crs` | Coordinate reference system | 3005 (BC Albers) |

### LiDAR Download

To download LiDAR from BC's Open Data Portal:

```r
download_config <- list(
  enabled    = TRUE,
  use_aoi    = TRUE,              # Use AOI shapefile
  most_recent = TRUE,             # Get most recent data
  download_dir = "D:/lidar"
)
```

## Outputs

| Output | Format | Description |
|--------|--------|-------------|
| DTM | GeoTIFF | Digital Terrain Model (ground elevation) |
| DSM | GeoTIFF | Digital Surface Model (top of canopy) |
| CHM | GeoTIFF | Canopy Height Model (DSM - DTM) |
| Slope | GeoTIFF | Terrain slope (degrees) |
| Aspect | GeoTIFF | Terrain aspect (degrees from north) |
| Trees | GeoPackage | Individual tree points with attributes |

## Data Sources

- **LiDAR**: [LidarBC Open Data](https://www2.gov.bc.ca/gov/content/data/geographic-data-services/lidarbc)
- **Processing**: [WhiteBox Tools](https://www.whiteboxgeo.com/)

## Author

BC Ministry of Forests - Omineca Region
