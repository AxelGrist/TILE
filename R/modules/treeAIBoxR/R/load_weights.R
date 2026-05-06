# load_weights.R
#
# Read a TreeAIBox JSON config + .pth state dict from disk, build the
# matching Segformer3D, and load weights with explicit key validation.

#' Default voxelization config bundle (used as a sanity reference)
#' @export
treeaibox_default_config <- function() {
  list(
    title = "branchcls",
    model = list(
      type   = "SegFormer",
      num_classes = 2L,                 # +1 in code => 3 (foliage/branch/stem)
      voxel_number_in_block  = c(112L, 112L, 112L),
      voxel_resolution_in_meter = c(0.04, 0.04, 0.04),
      patch_size  = 3L,
      decoder_dim = 64L,
      channel_dims = c(24L, 48L, 96L, 192L),
      SR_ratios = c(8L, 4L, 2L, 1L),
      num_heads = c(1L, 2L, 3L, 4L),
      MLP_ratios = c(2, 2, 2, 2),
      depths = c(1L, 1L, 4L, 1L),
      qkv_bias = TRUE
    )
  )
}

#' Strip non-tensor entries from a TreeAIBox state dict.
#' Upstream stores `max_accu` and `best_mIoU` alongside tensors.
#' @keywords internal
clean_state_dict <- function(sd) {
  drop <- c("max_accu", "best_mIoU")
  sd[setdiff(names(sd), drop)]
}

#' Compare expected vs actual state-dict keys and report mismatches.
#' @keywords internal
diagnose_state_dict <- function(model, state_dict) {
  expected <- names(model$state_dict())
  got      <- names(state_dict)
  missing  <- setdiff(expected, got)
  extra    <- setdiff(got, expected)
  if (length(missing) > 0L || length(extra) > 0L) {
    msg <- c(
      sprintf("State dict mismatch: %d missing, %d extra.",
              length(missing), length(extra)),
      if (length(missing) > 0L)
        paste("  Missing (in model, not in file):",
              paste(utils::head(missing, 8L), collapse = ", "),
              if (length(missing) > 8L) sprintf("... (+%d more)",
                                                 length(missing) - 8L) else ""),
      if (length(extra) > 0L)
        paste("  Extra (in file, not in model):",
              paste(utils::head(extra, 8L), collapse = ", "),
              if (length(extra) > 8L) sprintf("... (+%d more)",
                                               length(extra) - 8L) else "")
    )
    stop(paste(msg, collapse = "\n"), call. = FALSE)
  }
  invisible(TRUE)
}

#' Load a TreeAIBox model into R torch
#'
#' Accepts either a model name from \code{treeaibox_model_zoo()} (downloads
#' automatically on first use) or explicit \code{weights_path} /
#' \code{config_path} file paths.
#'
#' @param model_name    name from \code{treeaibox_model_zoo()}, e.g.
#'   \code{"woodcls_branch_tls_segformer3D_112_4cm(GPU2GBDistilled)"}.
#'   When supplied, \code{weights_path} and \code{config_path} are ignored.
#' @param weights_path  explicit path to a \code{.pth} file (used only when
#'   \code{model_name} is NULL)
#' @param config_path   explicit path to a \code{.json} file (used only when
#'   \code{model_name} is NULL)
#' @param device        \code{"cuda"} or \code{"cpu"} (default: auto-detect)
#' @param strict        if TRUE, missing/extra keys abort with an error;
#'   if FALSE, warnings only.
#' @param cache_dir     override the cache directory used by
#'   \code{ensure_model()} (ignored when explicit paths are supplied)
#' @return list with \code{model} (an nn_module in eval mode),
#'   \code{config}, \code{device}, \code{voxel_count},
#'   \code{voxel_res_m}, \code{num_classes_in}.
#' @export
load_treeaibox_model <- function(model_name   = NULL,
                                 weights_path = NULL,
                                 config_path  = NULL,
                                 device       = NULL,
                                 strict       = TRUE,
                                 cache_dir    = treeaibox_cache_dir()) {
  # Resolve paths: name takes priority over explicit paths
  if (!is.null(model_name)) {
    paths <- ensure_model(model_name, cache_dir = cache_dir)
    weights_path <- paths$pth
    config_path  <- paths$json
  }
  if (is.null(weights_path) || is.null(config_path))
    stop("Provide either 'model_name' or both 'weights_path' and 'config_path'.",
         call. = FALSE)
  if (!file.exists(weights_path))
    stop("Weights file not found: ", weights_path)
  if (!file.exists(config_path))
    stop("Config file not found: ", config_path)

  config <- jsonlite::fromJSON(config_path, simplifyVector = TRUE)
  cfg    <- config$model
  if (is.null(cfg)) stop("JSON config has no `model` block: ", config_path)

  # Resolve device
  if (is.null(device)) {
    device <- if (torch::cuda_is_available()) "cuda" else "cpu"
  }
  dev <- torch::torch_device(device)

  # Build architecture — dispatch on config$model$type + model_name suffix.
  # Upstream uses "ESegFormer" for vox3DESegFormer, "SegFormer" for v1.
  # Head type is inferred from the model name:
  #   *treeloc*  → detection_stem  (vox3DSegFormerDetection, if_stem=TRUE)
  #   *treeoff* / *crownoff* → regression (vox3DSegFormerRegression)
  #   otherwise  → segmentation (woodcls, stemcls, treefiltering, …)
  arch_type <- tolower(cfg$type %||% "segformer")
  nm_lower  <- tolower(model_name %||% basename(weights_path))
  head_type <- if (grepl("treeloc", nm_lower, fixed = TRUE)) {
    "detection_stem"
  } else if (grepl("treeoff|crownoff", nm_lower)) {
    "regression"
  } else {
    "segmentation"
  }
  # Load raw state dict early so we can inspect actual layer shapes when the
  # JSON omits num_classes (e.g. treeloc boreal model).
  raw <- torch::load_state_dict(weights_path)
  raw <- clean_state_dict(raw)

  # Resolve num_classes / out_chans.  When the JSON omits the field, infer it
  # from the final conv bias in the already-loaded state dict.
  nc_json <- if (!is.null(cfg$num_classes) && length(cfg$num_classes) > 0L) {
    as.integer(cfg$num_classes)
  } else if (head_type %in% c("detection_stem", "regression")) {
    # Peek at linear_pred.4.bias — length == out_channels of the final 1x1 conv
    bias_key <- "linear_pred.4.bias"
    if (!is.null(raw[[bias_key]])) {
      as.integer(raw[[bias_key]]$shape[1])
    } else {
      2L  # ultimate fallback
    }
  } else {
    2L  # segmentation models always have num_classes in the JSON
  }

  # For regression/detection models num_classes in the JSON refers to output
  # channels; no +1 offset applies — pass as-is.
  nc_override <- if (head_type %in% c("detection_stem", "regression")) {
    nc_json
  } else {
    NULL   # build_esegformer3d / build_segformer3d add +1 themselves
  }
  # treeoff uses a 2-channel input (occupancy + treeloc indicator broadcast to Z)
  in_chans_override <- if (head_type == "regression") 2L else 1L

  # Upstream TreeAIBox JSON configs often have "type": "SegFormer" even for
  # ESegFormer models — the type field is not reliably updated. Fall back to
  # checking the model name itself.
  use_eseg <- grepl("eseg", arch_type, fixed = TRUE) ||
              grepl("eseg", nm_lower,  fixed = TRUE)
  model <- if (use_eseg) {
    build_esegformer3d(config, num_classes_override = nc_override,
                       head_type  = head_type,
                       out_chans  = nc_json,
                       in_chans   = in_chans_override)
  } else {
    build_segformer3d(config)
  }
  model$to(device = dev)

  # (raw state dict was already loaded above for nc_json inspection)
  if (strict) {
    diagnose_state_dict(model, raw)
  } else {
    tryCatch(diagnose_state_dict(model, raw),
             error = function(e) warning(conditionMessage(e), call. = FALSE))
  }

  # `nn_module$load_state_dict()` is the R-torch idiom; if any keys still
  # mismatch (e.g. shape) it will raise here.
  model$load_state_dict(raw)
  model$eval()

  list(model = model, config = config, device = dev,
       voxel_count    = as.integer(cfg$voxel_number_in_block),
       voxel_res_m    = as.numeric(cfg$voxel_resolution_in_meter),
       num_classes_in = nc_json + if (head_type == "segmentation") 1L else 0L)
}
