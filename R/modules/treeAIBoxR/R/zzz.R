# zzz.R — package startup hooks

.onLoad <- function(libname, pkgname) {
  # Add the torch lib directory to PATH so Windows LoadLibrary can resolve
  # cuDNN sub-DLL dependencies (cudnn_engines_precompiled64_9.dll etc.)
  # when CUDA operations are first triggered.
  torch_pkg <- system.file(package = "torch")
  if (nzchar(torch_pkg)) {
    torch_lib <- normalizePath(file.path(torch_pkg, "lib"), mustWork = FALSE)
    if (dir.exists(torch_lib)) {
      old_path <- Sys.getenv("PATH")
      if (!grepl(torch_lib, old_path, fixed = TRUE)) {
        Sys.setenv(PATH = paste(torch_lib, old_path, sep = ";"))
      }
    }

    # Pre-load cuDNN DLLs that can be loaded directly (cudnn_graph64_9.dll
    # is not found by torch's runtime without an explicit prior dyn.load).
    for (dll in c("cudnn64_9.dll", "cudnn_graph64_9.dll",
                  "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll")) {
      path <- file.path(torch_lib, dll)
      if (file.exists(path)) {
        tryCatch(dyn.load(path, local = FALSE, now = FALSE),
                 error = function(e) invisible(NULL))
      }
    }
  }
  invisible(NULL)
}

.onAttach <- function(libname, pkgname) {
  # Download any missing models to the local cache.  Files that already exist
  # are skipped instantly, so this is a no-op after the first library() call.
  tryCatch(
    download_all_models(verbose = TRUE),
    error = function(e)
      packageStartupMessage(
        "[treeAIBoxR] Model download failed (no internet?): ",
        conditionMessage(e))
  )
  invisible(NULL)
}
