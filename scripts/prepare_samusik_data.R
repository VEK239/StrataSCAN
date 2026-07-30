#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(SummarizedExperiment))

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else "data/raw/samusik/Samusik_all_SE.rda"
output_root <- if (length(args) >= 2) args[[2]] else "data/processed"

load(input)
stopifnot(exists("d_SE_all"), nrow(d_SE_all) == 841644)

feature_info <- as.data.frame(colData(d_SE_all), stringsAsFactors = FALSE)
type_columns <- which(as.character(feature_info$marker_class) == "type")
stopifnot(length(type_columns) == 39)
feature_metadata <- data.frame(
  feature = as.character(feature_info$marker_name[type_columns]),
  marker_class = "type",
  channel_name = as.character(feature_info$channel_name[type_columns]),
  stringsAsFactors = FALSE
)
row_info <- as.data.frame(rowData(d_SE_all), stringsAsFactors = FALSE)
sample_ids <- sort(unique(as.character(row_info$sample_id)))
stopifnot(identical(sample_ids, sprintf("%02d", 1:10)))

for (sample_id in sample_ids) {
  keep <- as.character(row_info$sample_id) == sample_id
  sample_dir <- file.path(output_root, paste0("samusik_", sample_id))
  dir.create(sample_dir, recursive = TRUE, showWarnings = FALSE)

  features <- as.data.frame(assay(d_SE_all)[keep, type_columns, drop = FALSE])
  colnames(features) <- feature_metadata$feature
  features <- cbind(event_id = seq_len(nrow(features)), features)
  metadata <- data.frame(
    event_id = seq_len(sum(keep)),
    sample_id = sample_id,
    population_id = as.character(row_info$population_id[keep]),
    stringsAsFactors = FALSE
  )
  write.csv(features, gzfile(file.path(sample_dir, "features.csv.gz")), row.names = FALSE)
  write.csv(metadata, gzfile(file.path(sample_dir, "event_metadata.csv.gz")), row.names = FALSE)
  write.csv(feature_metadata, file.path(sample_dir, "feature_metadata.csv"), row.names = FALSE)
  writeLines(capture.output(sessionInfo()), file.path(sample_dir, "export_session.txt"))
  message("exported sample ", sample_id, ": ", sum(keep), " cells")
}
