rm(list=ls())

############################################################
# Template for final project                               #
############################################################

# Set your working directory here. 
setwd("...")

# Load necessary packages here. All packages must
# be available for install via install.packages()
# or remotes::install_github().

library("ROCR")
library("dplyr")

# define function to assess classification performance
auc <- function(phat,y){
	pred <- ROCR::prediction(phat, y)
	perf <- ROCR::performance(pred,"auc")
	auc <- perf@y.values[[1]]
	return(auc)
}

# load training data

load("AB4x_train.Rdata")

# Build your model; example code follows:

# ~~~ example code start ~~~
# Annina exploaratory analysis:

# to make sure that the original dataset is not altered
train_raw <- train  
train <- train_raw 

# missing variables: how many and where
na_counts <- colSums(is.na(train))
na_vars <- na_counts[na_counts > 0]
na_vars # missing values are in age and gender. 

# any non-numeric vars?
# Get the class of each column
var_classes <- sapply(train, class)
var_classes <- sapply(var_classes, `[`, 1)
table(var_classes)


# remove NAs
train_clean <- na.omit(train)
sum(is.na(train_clean))


#
# ~~~ example code end ~~~

# Save final model 
save(m, file="final_model.Rdata")