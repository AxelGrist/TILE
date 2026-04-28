# TILE module dispatcher
#
# Default behavior is ALS to preserve current workflow.
# Set environment variable TILE_MODULE=TLS to execute the TLS module scaffold.

module_choice <- toupper(trimws(Sys.getenv("TILE_MODULE", unset = "ALS")))

resolve_module_path <- function(filename) {
  candidates <- c(
    file.path("R", "modules", filename),
    file.path("modules", filename)
  )

  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0) {
    stop("Module not found: ", filename)
  }

  existing[[1]]
}

als_module_path <- resolve_module_path("als_module.R")
tls_module_path <- resolve_module_path("tls_module.R")

if (module_choice == "ALS") {
  source(als_module_path)
} else if (module_choice == "TLS") {
  source(tls_module_path)

  # Run TLS scaffold with defaults unless user has already defined tls_config.
  if (exists("tls_config", inherits = TRUE)) {
    run_tls_module(tls_config)
  } else {
    run_tls_module()
  }
} else {
  stop("Unknown TILE_MODULE value: ", module_choice, ". Use ALS or TLS.")
}
