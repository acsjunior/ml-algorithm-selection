require(tidyverse)
require(caret)
require(e1071)
require(pROC)
require(randomForest)
require(gbm)
require(adabag)
require(xgboost)
require(RSNNS)
require(lda)
require(kernlab)
require(C50)
SEED <- 1986

# --------------------------------------------------------------------------
# Preparing data:

df_train <- read.csv("data/anonymous_train.csv")
df_test <- read.csv("data/anonymous_test.csv")
df <- rbind(df_train, df_test)
numeric_vars <- c("x2", "x4", "x5", "x6", "x9", "x22", "x27", "x29", "x30", "x31")
dummy_vars <- c("x11", "x13", "x14", "x20")
model_vars <- c("y", numeric_vars, dummy_vars)

df <- df[model_vars]
df$y <- factor(df$y)
#levels(df$y) <- c("false", "true")
#names(df) <- c("y", paste0("x_",1:14))

# --------------------------------------------------------------------------
n_folds <- 10
train_ctrl <- trainControl(method = "cv", number = n_folds)
df_final <- NULL

# --------------------------------------------------------------------------
# Logistic regression:

model_label <- "rlog"

set.seed(SEED)
rlog_model <- caret::train(form = y ~ .,
               data = df,
               method = "glm",
               family = "binomial",
               trControl = train_ctrl)

model <- rlog_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))


tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}

# --------------------------------------------------------------------------
# Random forest:


model_label <- "rfor"

set.seed(SEED)
rfor_model <- caret::train(form = y ~ .,
                           data = df,
                           method = "rf",
                           trControl = train_ctrl)

model <- rfor_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}

# --------------------------------------------------------------------------
# Gradient Boosting Machine Estocástico


model_label <- "gbm"

set.seed(SEED)
gbm_model <- caret::train(form = y ~ .,
                           data = df,
                           method = "gbm",
                           verbos = F,
                           trControl = train_ctrl)

model <- gbm_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}


# --------------------------------------------------------------------------
# XG Boost


model_label <- "xgb"

set.seed(SEED)
xgb_model <- caret::train(form = y ~ .,
                          data = df,
                          method = "xgbTree",
                          verbos = F,
                          trControl = train_ctrl)

model <- xgb_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}



# --------------------------------------------------------------------------
# Multi Layer Perceptron


model_label <- "mlp"

set.seed(SEED)
mlp_model <- caret::train(form = y ~ .,
                          data = df,
                          method = "mlp",
                          trControl = train_ctrl)

model <- mlp_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}


# --------------------------------------------------------------------------
# LDA


model_label <- "lda"

set.seed(SEED)
lda_model <- caret::train(form = y ~ .,
                          data = df,
                          method = "lda",
                          trControl = train_ctrl)

model <- lda_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}


# --------------------------------------------------------------------------
# KNN


model_label <- "knn"

set.seed(SEED)
knn_model <- caret::train(form = y ~ .,
                          data = df,
                          method = "knn",
                          trControl = train_ctrl)

model <- knn_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}


# --------------------------------------------------------------------------
# SVM Radial


model_label <- "svm"

set.seed(SEED)
svm_model <- caret::train(form = y ~ .,
                          data = df,
                          method = "svmRadial",
                          trControl = train_ctrl)

model <- svm_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}


# --------------------------------------------------------------------------
# C5.0


model_label <- "c50"

set.seed(SEED)
c50_model <- caret::train(form = y ~ .,
                          data = df,
                          method = "C5.0",
                          trControl = train_ctrl)

model <- c50_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}


# --------------------------------------------------------------------------
# Ada Boosting M1


model_label <- "ada"

set.seed(SEED)
ada_model <- caret::train(form = y ~ .,
                          data = df,
                          method = "AdaBoost.M1",
                          trControl = train_ctrl)

model <- ada_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}


# --------------------------------------------------------------------------
# QDA


model_label <- "qda"

set.seed(SEED)
qda_model <- caret::train(form = y ~ .,
                          data = df,
                          method = "qda",
                          trControl = train_ctrl)

model <- qda_model
time <- as.numeric(model$times$everything[1]) / n_folds
pred <- predict(model)
cm <- caret::confusionMatrix(data = pred, reference = df$y, positive = "1", mode = "everything")
rocobj <- roc(as.numeric(as.character(df$y)), as.numeric(as.character(pred)), plot=F, ci=T, ci.sp = T)
auc <- as.numeric(auc(rocobj))

tn <- cm$table[1]
fp <- cm$table[2]
fn <- cm$table[3]
tp <- cm$table[4]

values <- data.frame(tn, tp, fn, fp)
folds <- data.frame(n_folds)
avg_time <- data.frame(time)
metrics1 <- t(as.data.frame(cm$overall))
metrics2 <- t(as.data.frame(cm$byClass))
row.names(values) <- model_label
row.names(folds) <- model_label
row.names(avg_time) <- model_label
row.names(metrics1) <- model_label
row.names(metrics2) <- model_label

concat <- cbind(values, folds, avg_time, metrics1, metrics2)

if (is.null(df_final)) {
  df_final <- concat
} else {
  df_final <- rbind(df_final, concat)
}

# --------------------------------------------------------------------------
# Export

write.csv(df_final, "metrics.csv")
