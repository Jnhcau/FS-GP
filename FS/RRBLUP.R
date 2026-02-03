library(rrBLUP)

## =============== Load data ===============
train_file <- "rrblup_data/ENet_RF/r1_f1/train.csv"
test_file  <- "rrblup_data/ENet_RF/r1_f1/test.csv"

train <- read.csv(train_file, check.names = FALSE)
test  <- read.csv(test_file,  check.names = FALSE)

id_train <- train[, 1]
id_test  <- test[, 1]

pheno_train <- train$y
pheno_test  <- test$y  

geno_train <- as.matrix(train[, -c(1, ncol(train))])
geno_test  <- as.matrix(test[,  -c(1, ncol(test))])
## =============== rrBLUP ===============
model <- mixed.solve(
  y = pheno_train,
  Z = geno_train
)

u    <- as.matrix(model$u)
beta <- as.numeric(model$beta)

## =============== 预测 ===============
pred_test <- geno_test %*% u + beta
pred_test <- as.numeric(pred_test)

## =============== 评估 ===============
pcc <- cor(pred_test, pheno_test, use = "complete.obs")

cat("Test PCC =", round(pcc, 4), "\n")

## =============== 保存结果 ===============
out <- data.frame(
  ID = id_test,
  y_true = pheno_test,
  y_pred = pred_test
)

write.csv(out, "rrblup_prediction.csv", row.names = FALSE)
