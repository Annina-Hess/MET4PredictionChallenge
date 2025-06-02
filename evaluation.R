# version 25 May 2021

############################################################
# Template for final project                               #
############################################################

rm(list=ls())

# Set your working directory here. 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load necessary packages here. All packages must
# be available for install via install.packages()
# or remotes::install_github().

library(ROCR)
library(dplyr)
library(tidymodels)
library(xgboost)
library(doFuture)
library(vip)
library(ggplot2)


# define function to assess classification performance

auc <- function(phat,y){
  pred <- ROCR::prediction(phat, y)
  perf <- ROCR::performance(pred,"auc")
  auc <- perf@y.values[[1]]
  return(auc)
}

# load train and test data

load("AB4x_train.Rdata")
load("AB4x_eval_mock.Rdata")
# The mock data will be replaced by the 
# hold-out data for final evaluation.

# Include all necessary preprocessing steps
# that should be applied to the test data.

# ~~~ example code start ~~~

test$emig <- factor(test$emig, levels = c("No", "Yes"))

# ~~~ example code end ~~~

# Load trained model 
load("final_model.Rdata")

# Obtain predictions
# Note that `pred` should be a vector of predicted probabilities;
# not a vector of predicted classes
# ~~~ example code start ~~~
pred <- predict(m, new_data = test, type = "prob")$.pred_Yes

# ~~~ example code end ~~~

# Model performance:
truth <- test$emig

cat("The final score:","\n") 
auc(pred,truth)
