# download_model.R
#
# On-demand model cache for treeAIBoxR.
#
# Weights (.pth) are downloaded from the GitHub v1.0 release assets.
# Config (.json) files are downloaded from the repo source tree.
#
# Cache location (user-writable, persists across R sessions):
#   tools::R_user_dir("treeAIBoxR", "cache")  — typically
#   %LOCALAPPDATA%\R\cache\R\treeAIBoxR  on Windows

# Base URLs ---------------------------------------------------------------
.RELEASE_BASE <- "https://github.com/NRCan/TreeAIBox/releases/download/v1.0"
.RAW_FILTER   <- "https://raw.githubusercontent.com/NRCan/TreeAIBox/main/modules/filter"
.RAW_ISONET   <- "https://raw.githubusercontent.com/NRCan/TreeAIBox/main/modules/treeisonet"

# Model zoo (mirrors model_zoo.json from the repo) ------------------------

#' List all available TreeAIBox model names
#'
#' These names match the model_zoo.json from the upstream repo.  Pass any
#' one of them to \code{ensure_model()} or \code{load_treeaibox_model()}.
#'
#' @return character vector of model names
#' @export
treeaibox_model_zoo <- function() {
  c(
    # --- TreeFiltering ---
    "treefiltering_tls_esegformer3D_128_8cm(GPU3GB)",
    "treefiltering_als_esegformer3D_128_15cm(GPU3GB)",
    "treefiltering_als_esegformer3D_128_50cm(GPU3GB)",
    "treefiltering_als_esegformer3D_128_80cm(GPU3GB)",
    "treefiltering_uav_esegformer3D_128_12cm(GPU3GB)",
    # --- Urban filtering ---
    "urbanfiltering_als_esegformer3D_112_30cm(GPU3GB)",
    # --- TreeisoNet ---
    "treeisonet_als_reclamation_treeloc_esegformer3D_128_10cm(GPU4GB)",
    "treeisonet_als_reclamation_treeoff_esegformer3D_128_10cm(GPU4GB)",
    "treeisonet_tls_boreal_stemcls_esegformer3D_128_4cm(GPU3GB)",
    "treeisonet_tls_boreal_stemcls_esegformer3D_128_10cm(GPU3GB)",
    "treeisonet_tls_boreal_treeloc_esegformer3D_128_10cm(GPU3GB)",
    "treeisonet_tls_boreal_crownoff_esegformer3D_128_15cm(GPU4GB)",
    "treeisonet_uav_mixedwood_stemcls_esegformer3D_128_8cm(GPU3GB)",
    "treeisonet_uav_mixedwood_treeloc_esegformer3D_128_10cm(GPU3GB)",
    "treeisonet_uav_mixedwood_crownoff_esegformer3D_128_15cm(GPU4GB)",
    # --- WoodCls (stems) ---
    "woodcls_stem_tls_esegformer3D_128_4cm(GPU3GB)",
    "woodcls_stem_tls_esegformer3D_128_10cm(GPU3GB)",
    "woodcls_stem_tls_segformer3D_112_4cm(GPU12GB)",
    "woodcls_stem_tls_segformer3D_112_20cm(GPU8GB)",
    # --- WoodCls (branches) ---
    "woodcls_branch_tls_esegformer3D_128_2.5cm(GPU3GB)",
    "woodcls_branch_tls_segformer3D_112_4cm(GPU2GBDistilled)",
    "woodcls_branch_tls_segformer3D_112_4cm(GPU6GB)"
  )
}

# Helpers -----------------------------------------------------------------

# The release asset filenames replace "(" with "_" and drop ")".
# e.g.  woodcls_branch_tls_segformer3D_112_4cm(GPU2GBDistilled)
#    -> woodcls_branch_tls_segformer3D_112_4cm_GPU2GBDistilled
.pth_stem <- function(name) {
  gsub(")", "", gsub("(", "_", name, fixed = TRUE), fixed = TRUE)
}

# JSON configs live in the repo source tree, filename keeps the parens.
.json_subdir <- function(name) {
  if (grepl("^treeisonet_", name)) .RAW_ISONET else .RAW_FILTER
}

#' Return (and create if needed) the treeAIBoxR model cache directory
#'
#' The default is \code{tools::R_user_dir("treeAIBoxR", "cache")}.
#' Override by setting the environment variable \code{TREEAIBOXR_CACHE}.
#'
#' @return path string
#' @export
treeaibox_cache_dir <- function() {
  d <- Sys.getenv("TREEAIBOXR_CACHE", unset = "")
  if (nchar(d) == 0L)
    d <- tools::R_user_dir("treeAIBoxR", "cache")
  if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

# Safe download with progress and retry ----------------------------------
.download_once <- function(url, dest, verbose = TRUE) {
  if (file.exists(dest)) return(invisible(dest))
  tmp <- paste0(dest, ".part")
  on.exit(if (file.exists(tmp)) file.remove(tmp), add = TRUE)
  if (verbose) message(sprintf("  Downloading %s\n    -> %s", url, dest))
  tryCatch(
    utils::download.file(url, tmp, mode = "wb", quiet = !verbose),
    error = function(e)
      stop(sprintf("Failed to download:\n  %s\n  %s", url,
                   conditionMessage(e)), call. = FALSE)
  )
  file.rename(tmp, dest)
  invisible(dest)
}

# Public API --------------------------------------------------------------

#' Ensure a TreeAIBox model is in the local cache (download if missing)
#'
#' Checks \code{treeaibox_cache_dir()} for the \code{.pth} and \code{.json}
#' files.  Downloads from the GitHub v1.0 release / source tree if either is
#' absent.
#'
#' @param name  one of the strings returned by \code{treeaibox_model_zoo()}
#' @param cache_dir  override the default cache directory
#' @param verbose  print download progress (default TRUE)
#'
#' @return named list with \code{pth} and \code{json} file paths
#' @export
ensure_model <- function(name,
                         cache_dir = treeaibox_cache_dir(),
                         verbose   = TRUE) {
  zoo <- treeaibox_model_zoo()
  if (!name %in% zoo)
    stop(sprintf(
      "Unknown model '%s'.\nRun treeaibox_model_zoo() to see valid names.",
      name), call. = FALSE)

  pth_file  <- file.path(cache_dir, paste0(.pth_stem(name), ".pth"))
  json_file <- file.path(cache_dir, paste0(name, ".json"))

  pth_url  <- paste0(.RELEASE_BASE, "/", .pth_stem(name), ".pth")
  json_url <- paste0(.json_subdir(name), "/", utils::URLencode(paste0(name, ".json")))

  if (verbose && (!file.exists(pth_file) || !file.exists(json_file)))
    message(sprintf("[treeAIBoxR] Caching model '%s' in:\n  %s", name, cache_dir))

  .download_once(pth_url,  pth_file,  verbose = verbose)
  .download_once(json_url, json_file, verbose = verbose)

  list(pth = pth_file, json = json_file)
}

#' Download all TreeAIBox models to the local cache
#'
#' Iterates over every name in \code{treeaibox_model_zoo()} and calls
#' \code{ensure_model()} for each.  Files that already exist are skipped
#' instantly, so this is safe to call repeatedly (e.g. from \code{.onAttach}).
#'
#' @param cache_dir override the default cache directory
#' @param verbose   print download progress (default TRUE)
#'
#' @return invisible named list of \code{list(pth, json)} per model
#' @export
download_all_models <- function(cache_dir = treeaibox_cache_dir(),
                                verbose   = TRUE) {
  zoo <- treeaibox_model_zoo()
  # Check upfront whether anything is actually missing to avoid spamming the
  # user on every library() when everything is already cached.
  missing <- vapply(zoo, function(nm) {
    pth  <- file.path(cache_dir, paste0(.pth_stem(nm), ".pth"))
    json <- file.path(cache_dir, paste0(nm, ".json"))
    !file.exists(pth) || !file.exists(json)
  }, logical(1L))

  if (any(missing)) {
    if (verbose)
      message(sprintf(
        "[treeAIBoxR] Downloading %d missing model(s) to:\n  %s",
        sum(missing), cache_dir))
    results <- list()
    failed  <- character(0L)
    for (nm in zoo[missing]) {
      res <- tryCatch(
        ensure_model(nm, cache_dir = cache_dir, verbose = verbose),
        error = function(e) {
          message(sprintf("[treeAIBoxR] Skipping '%s': %s", nm,
                          conditionMessage(e)))
          NULL
        }
      )
      if (is.null(res)) failed <- c(failed, nm) else results[[nm]] <- res
    }
    if (length(failed) > 0L)
      warning(sprintf(
        "[treeAIBoxR] %d model(s) could not be downloaded (no internet / 404):\n  %s",
        length(failed), paste(failed, collapse = "\n  ")),
        call. = FALSE)
  } else {
    results <- list()
  }
  invisible(results)
}
