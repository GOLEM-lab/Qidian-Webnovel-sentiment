#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Pinned R environment for the GAM analysis.
#
# These versions are not a wish list: they are what sessionInfo() recorded at
# the end of the run that produced GAM/GAM_Model_Comparison.pdf. The record is
# the tail of GAM/gam_habrok_output.txt, from R 4.5.1 (2025-06-13) on
# AlmaLinux 8.10 with FlexiBLAS/OpenBLAS.
#
# Pinning matters here because CRAN has moved on: as of this writing lme4 is
# at 2.1-0 against the 1.1-38 used below — a major version — and performance
# has gone from 0.15.3 to 0.19.0. An unpinned re-run is not the same analysis.
#
# renv is the better tool for this and is what the README recommends. Use this
# script when a project library is awkward, e.g. on a cluster where R comes
# from a module.
#
# Usage:
#   Rscript scripts/install_r_packages.R            # install what is missing
#   Rscript scripts/install_r_packages.R --check    # report only, install nothing
#   Rscript scripts/install_r_packages.R --lib=/path/to/library
# ---------------------------------------------------------------------------

PINS <- c(
  # tied to the R version — see the note below
  nlme          = "3.1-168",
  Matrix        = "1.7-4",
  mgcv          = "1.9-4",
  # modelling
  lme4          = "1.1-38",
  performance   = "0.15.3",
  itsadug       = "2.5",
  plotfunctions = "1.5",
  # data handling and figures
  dplyr         = "1.1.4",
  readr         = "2.1.6",
  ggplot2       = "4.0.2",
  patchwork     = "1.3.2"
)

args     <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args
lib_arg  <- grep("^--lib=", args, value = TRUE)
lib      <- if (length(lib_arg)) sub("^--lib=", "", lib_arg[1]) else .libPaths()[1]

if (!nzchar(getOption("repos")["CRAN"]) || getOption("repos")["CRAN"] == "@CRAN@") {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

cat("R:       ", R.version.string, "\n")
cat("library: ", lib, "\n")
cat("recorded under R 4.5.1; a different R will resolve base packages differently\n\n")

installed_version <- function(p) {
  v <- tryCatch(as.character(utils::packageVersion(p, lib.loc = .libPaths())),
                error = function(e) NA_character_)
  v
}

status <- data.frame(package = names(PINS), pinned = unname(PINS),
                     installed = vapply(names(PINS), installed_version, ""),
                     stringsAsFactors = FALSE)
# as.character(packageVersion()) renders "3.1-168" as "3.1.168", so compare
# through numeric_version, which treats "-" and "." as the same separator.
same <- function(a, b) !is.na(a) && numeric_version(a) == numeric_version(b)
status$state <- vapply(seq_len(nrow(status)), function(i)
  if (is.na(status$installed[i])) "missing"
  else if (same(status$installed[i], status$pinned[i])) "ok"
  else "wrong version", character(1))
print(status, row.names = FALSE)
cat("\n")

if (check_only) {
  bad <- sum(status$state != "ok")
  cat(if (bad == 0) "All pinned versions present.\n"
      else sprintf("%d package(s) differ from the pinned versions.\n", bad))
  quit(status = if (bad == 0) 0 else 1)
}

if (!requireNamespace("remotes", quietly = TRUE)) {
  cat("installing remotes (needed to fetch specific versions)\n")
  utils::install.packages("remotes", lib = lib)
}

todo <- status[status$state != "ok", ]
if (nrow(todo) == 0) {
  cat("Nothing to do.\n")
} else {
  for (i in seq_len(nrow(todo))) {
    p <- todo$package[i]; v <- todo$pinned[i]
    cat(sprintf("installing %s %s ...\n", p, v))
    remotes::install_version(p, version = v, lib = lib,
                             repos = getOption("repos"), upgrade = "never")
  }
}

cat("
Notes
-----
* nlme, Matrix and mgcv ship with R itself. Installing other versions over a
  module-provided R can fail or be silently ignored, and the version you get
  is in practice decided by the R you load. If they come out different from
  the pins above, record the R version rather than forcing these three.
* lme4 is compiled against Matrix's ABI. If you reinstall Matrix from source,
  reinstall lme4 from source too, or lme4 will fail at load with a message
  about Matrix ABI mismatch.
* This pins direct dependencies only. For the full transitive set, including
  the 44 packages loaded via namespace, use renv (see README).
")
