# registry.R builds a registry fit's design from its documented formula, as lme4 builds it,
# or refits a formula as mixedbench's lme4.R fits it, with the max |eta| lme4.R leaves out.
#
# Usage: Rscript registry.R design|refit <dir> <lme4.R>
# <dir> holds table.csv and input.json, as mixedbench's frequentist.fit writes them,
# and <lme4.R> is mixedbench's own script, whose definitions registry.R loads as they are.
# registry.R writes back into <dir>:
#   design.json  (design) the rows lme4 keeps (1-based), X and its column names, the offset
#                (0 without one), and per factor in lme4's order its name, terms and levels,
#                each row's random columns and each row's level (1-based)
#   output.json  (refit) what lme4.R writes
#   eta.json     (refit) the max |eta| over the rows, where the fit has estimates
#
# pipeline:
# ---------
# <dir>
# -> lme4.R's definitions                                loadLme4
# -> the table, its categorical columns coded           readTable (lme4.R's)
# design:
# -> lme4's model frame, X and random terms             buildDesign
# -> per factor, each row's random columns and level    describeFactors
# -> design.json
# refit:
# -> lme4.R's fit, its description with max |eta|       fitWithin (lme4.R's), describeWithEta
# -> output.json, eta.json

# =====================================================================================
# the steps
# =====================================================================================

# every definition of mixedbench's lme4.R, lme4 attached: its expressions before its main block,
# which opens on sessionInfo()
loadLme4 <- function(path) {
  expressions <- parse(path, keep.source = FALSE)
  main <- NULL
  for (i in seq_along(expressions)) {
    if (identical(expressions[[i]], quote(print(sessionInfo())))) main <- c(main, i)
  }
  stopifnot(length(main) == 1)
  for (i in seq_len(main - 1)) eval(expressions[[i]], globalenv())
}

# lmer's and glmer's own parse of the formula, with their checks and the columns they drop
buildDesign <- function(table, input) {
  formula <- as.formula(input$formula)
  if (input$family == "gaussian") {
    return(lFormula(formula, data = table))
  }
  glFormula(formula, data = table, family = FAMILY[[input$family]])
}

# lme4 splits `(1 + x || g)` into two terms of g, so a factor's columns are its terms' side by side.
# A term's columns per row are read off its Zt, which holds per level its columns in turn
describeFactors <- function(reTrms) {
  assign <- attr(reTrms$flist, "assign")  # each term's factor
  factors <- list()
  for (k in seq_along(reTrms$flist)) {
    levels <- nlevels(reTrms$flist[[k]])
    z <- NULL
    for (i in which(assign == k)) {
      columns <- length(reTrms$cnms[[i]])
      # a row lies in one level, so the sum over levels leaves the row's own columns
      sum_levels <- kronecker(matrix(1, levels, 1), diag(columns))
      z <- cbind(z, as.matrix(t(reTrms$Ztlist[[i]]) %*% sum_levels))
    }
    factors[[k]] <- list(
      name = names(reTrms$flist)[k], terms = sum(assign == k), levels = levels,
      z = unname(z), codes = as.integer(reTrms$flist[[k]])
    )
  }
  factors
}

# lme4.R's description of a fit, with the max |eta| over the rows (the offset in it, as in sd_eta)
describeWithEta <- function(model, table, family) {
  eta <- predict(model, type = "link")
  c(describeLme4(model, table, family), list(max_abs_eta = max(abs(eta))))
}

# =====================================================================================
# main
# =====================================================================================

args <- commandArgs(trailingOnly = TRUE)
mode <- args[[1]]
stopifnot(mode %in% c("design", "refit"))
dir <- args[[2]]
loadLme4(args[[3]])
input <- jsonlite::read_json(file.path(dir, "input.json"))

if (mode == "design") {
  table <- readTable(dir, input)
  design <- buildDesign(table, input)
  offset <- model.offset(design$fr)
  if (is.null(offset)) offset <- rep(0, nrow(design$fr))
  output <- list(
    rows = as.integer(rownames(design$fr)),
    columns = I(as.character(colnames(design$X))),  # none for `0 + (1 | g)`
    X = matrix(design$X, nrow(design$X)),
    offset = offset,
    factors = describeFactors(design$reTrms)
  )
  jsonlite::write_json(output, file.path(dir, "design.json"), auto_unbox = TRUE, digits = NA)
} else {
  # first, so that R.log names the R and lme4 versions, as lme4.R's own does
  print(sessionInfo())
  table <- readTable(dir, input)
  # lme4.R's fit within its limit, which describes the fit by the name describeFit
  describeLme4 <- describeFit
  describeFit <- describeWithEta
  output <- c(fitWithin(table, input), list(lme4 = as.character(packageVersion("lme4"))))
  # max |eta| in a file of its own, since mixedbench reads output.json into its fixed fields
  if (!is.null(output[["max_abs_eta"]])) {
    eta <- list(max_abs_eta = output[["max_abs_eta"]])
    jsonlite::write_json(eta, file.path(dir, "eta.json"), auto_unbox = TRUE, digits = NA)
  }
  output[["max_abs_eta"]] <- NULL
  jsonlite::write_json(
    output, file.path(dir, "output.json"), auto_unbox = TRUE, digits = NA, na = "null"
  )
}
