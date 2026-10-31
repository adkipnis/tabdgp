# registry.R builds a registry fit's design from its documented formula, as lme4 builds it.
#
# Usage: Rscript registry.R <dir>
# <dir> holds table.csv and input.json, as mixedbench's frequentist.fit writes them.
# registry.R writes back into <dir>:
#   design.json  the rows lme4 keeps (1-based), X and its column names, the offset (0 without one),
#                and per factor in lme4's order its name, terms and levels,
#                each row's random columns and each row's level (1-based)
#
# pipeline:
# ---------
# <dir>
# -> the table, its categorical columns coded         readTable
# -> lme4's model frame, X and random terms           buildDesign
# -> per factor, each row's random columns and level  describeFactors
# -> design.json

suppressPackageStartupMessages(library(lme4))

FAMILY <- list(bernoulli = binomial(), poisson = poisson())

# =====================================================================================
# the steps
# =====================================================================================

# copied from mixedbench 1.0.12's frequentist/lme4.R, so that the design reads the table
# as the refit will:
# a categorical column is read as text, then coded by its levels;
# a value that is no level would be lost as a missing value
readTable <- function(dir, input) {
  columns <- names(input$levels)
  classes <- setNames(rep("character", length(columns)), columns)
  table <- read.csv(
    file.path(dir, "table.csv"), check.names = FALSE, colClasses = classes, na.strings = ""
  )
  for (column in columns) {
    levels <- unlist(input$levels[[column]])
    coded <- factor(table[[column]], levels = levels, ordered = column %in% unlist(input$ordered))
    stopifnot(identical(is.na(coded), is.na(table[[column]])))
    table[[column]] <- coded
  }
  table
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

# =====================================================================================
# main
# =====================================================================================

dir <- commandArgs(trailingOnly = TRUE)[[1]]
input <- jsonlite::read_json(file.path(dir, "input.json"))
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
