# Load Peptides library
install.packages("Peptides")  # only if needed
library(Peptides)

# Read peptides from file for NPP
peptides <- readLines("NPP.txt")

# for PPP
peptide_table <- read.csv("PPP.csv", header = TRUE, stringsAsFactors = FALSE)
peptides <- peptide_table$PEPTIDE.SEQUENCE

# Function to compute mean Z-scale vector for any peptide
compute_mean_zscale <- function(peptide) {
  z_list <- zScales(peptide)  # returns a list
  
  # Convert list of vectors → matrix
  z_matrix <- do.call(rbind, z_list)
  
  # Compute column means
  colMeans(z_matrix)
}

# Apply to all peptides
zscale_matrix <- t(vapply(peptides, compute_mean_zscale, FUN.VALUE = numeric(5)))

# Add column names
colnames(zscale_matrix) <- c("Z1", "Z2", "Z3", "Z4", "Z5")

# Add peptide as first column
zscale_df <- as.data.frame(zscale_matrix)
zscale_df$peptide <- peptides

# Save result to CSV
write.csv(zscale_df, "PPP_tscale.csv", row.names = FALSE)
