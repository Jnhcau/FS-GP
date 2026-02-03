library(rrBLUP)

#From Prediction.py
base_dir <- "rrblup_data/ENet_RF"

fold_dirs <- list.dirs(base_dir, recursive = TRUE, full.names = TRUE)
fold_dirs <- fold_dirs[fold_dirs != base_dir]

all_results <- list()
perf <- data.frame()

for (d in fold_dirs) {
  
  cat("Processing:", d, "\n")
  
  train_file <- file.path(d, "train.csv")
  test_file  <- file.path(d, "test.csv")
  
  if (!file.exists(train_file) | !file.exists(test_file)) {
    cat("  Skip: files not found\n")
    next
  }
  
  ## ========== Load ==========
  train <- read.csv(train_file, check.names = FALSE)
  test  <- read.csv(test_file,  check.names = FALSE)
  
  id_train <- train[, 1]
  id_test  <- test[, 1]
  
  pheno_train <- train$y
  pheno_test  <- test$y  
  
  geno_train <- as.matrix(train[, -c(1, ncol(train))])
  geno_test  <- as.matrix(test[,  -c(1, ncol(test))])
  
  ## ========== rrBLUP ==========
  model <- mixed.solve(
    y = pheno_train,
    Z = geno_train
  )
  
  u    <- as.matrix(model$u)
  beta <- as.numeric(model$beta)

  pred_test <- as.numeric(geno_test %*% u + beta)
  pcc <- cor(pred_test, pheno_test, use = "complete.obs")
  out <- data.frame(
    ID = id_test,
    y_true = pheno_test,
    y_pred = pred_test
  )
  
  write.csv(
    out,
    file.path(d, "rrblup_prediction.csv"),
    row.names = FALSE
  )
  
  ## ========== Collect ==========
  perf <- rbind(
    perf,
    data.frame(
      Fold = basename(d),
      PCC = pcc
    )
  )
  
  cat("  PCC =", round(pcc, 4), "\n")
}

## ========== Save summary ==========
write.csv(perf, "rrblup_PCC_summary.csv", row.names = FALSE)

cat("Finished! Mean PCC =",
    round(mean(perf$PCC), 4), "\n")
