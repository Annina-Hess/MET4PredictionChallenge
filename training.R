rm(list=ls())

############################################################
# Template for final project                               #
############################################################

# Set your working directory here. 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load necessary packages here. All packages must
# be available for install via install.packages()
# or remotes::install_github().

library("ROCR")
library("dplyr")
library(tidymodels)

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

# ~~~ PREPROCESSING ~~~

# to make sure that the original dataset is not altered
train_raw <- train  
train <- train_raw 

# missing variables: how many and where
na_counts <- colSums(is.na(train))
na_vars <- na_counts[na_counts > 0]
na_vars # missing values are in age and gender. Instead of removing, I impute missing values in the recipes

# any non-numeric vars?
# Get the class of each column
var_classes <- sapply(train, class)
var_classes <- sapply(var_classes, `[`, 1)
table(var_classes)

#"emig" is character
train$emig <- factor(train$emig, levels = c("No", "Yes"))

# ~~~ ANNINA EXPLORATORY ANALYSIS ~~~

# ~~~ Logistic Regression ~~~
#tidymodels is imported above
set.seed(123)  

folds <- vfold_cv(train, v = 3) # should be increased

# Define a preprocessing recipe
logistic_recipe <- recipe(emig ~ ., data = train) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_mode(all_nominal_predictors())%>%
  step_dummy(all_nominal_predictors()) %>%  # Convert categorical predictors to dummy variables
  step_zv(all_predictors()) %>% 
  step_lincomb(all_predictors()) %>% 
  step_corr(all_predictors(), threshold = 0.9)  # Remove highly correlated predictors# Remove predictors with zero variance (i.e., same value for all rows)

# Specify a logistic regression model
logistic_spec <- logistic_reg() %>%
  set_engine("glm") %>%           # Use base R's GLM engine
  set_mode("classification")      # Explicitly declare classification mode

# Combine recipe and model into a workflow
logistic_wf <- workflow() %>%
  add_model(logistic_spec) %>%
  add_recipe(logistic_recipe)

# Fit model with 10-fold cross-validation
cv_results <- fit_resamples(
  logistic_wf,
  resamples = folds,
  metrics = metric_set(accuracy, roc_auc, sensitivity, specificity),
  control = control_resamples(save_pred = TRUE)
)

best_model <- select_best(cv_results, metric = "roc_auc")

final_logistic_wf <- finalize_workflow(logistic_wf, best_model)

final_logistic_reg <- fit(final_logistic_wf, data = train)

save(final_logistic_reg, file = "final_logistic_model.Rdata")

# ~~~ Elastic Net Regression ~~~

elastic_recipe <- recipe(emig ~ ., data = train) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_corr(all_predictors(), threshold = 0.9)

# 3. Elastic Net model specification
elastic_spec <- logistic_reg(
  penalty = tune(),   # We'll tune this
  mixture = 0.5       # 0.5 = Elastic Net (mix of Lasso and Ridge)
) %>%
  set_engine("glmnet") %>%
  set_mode("classification") # hallo

# 4. Workflow
elastic_wf <- workflow() %>%
  add_recipe(elastic_recipe) %>%
  add_model(elastic_spec)

# 5. Penalty tuning grid
penalty_grid <- grid_regular(penalty(), levels = 30)

# 6. Tune model
cv_results <- tune_grid(
  elastic_wf,
  resamples = folds,
  grid = penalty_grid,
  metrics = metric_set(roc_auc, accuracy),
  control = control_grid(save_pred = TRUE)
)

# 7. Select best penalty
best_model <- select_best(cv_results, metric = "roc_auc")

# 8. Finalize workflow with best model
final_wf <- finalize_workflow(elastic_wf, best_model)

# 9. Fit to full training data
final_elasticnet_reg <- fit(final_wf, data = train)

# Save final model 
save(final_elasticnet_reg, file="final_elasticnet_model.Rdata")

# ~~~ Random Forest ~~~

rf_recipe <- recipe(emig ~ ., data = train) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_corr(all_predictors(), threshold = 0.9)

# 2. Random Forest model specification (tunable)
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500  # number of trees to grow
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# 3. Workflow
rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_spec)

# 4. Create tuning grid
rf_grid <- grid_regular(
  mtry(range = c(2, 10)),    # you can adjust depending on number of predictors
  min_n(range = c(2, 10)),
  levels = 5
)

# 5. Cross-validation
set.seed(123)
cv_results_rf <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc, accuracy),
  control = control_grid(save_pred = TRUE)
)

# 6. Select best parameters
best_rf <- select_best(cv_results_rf, metric = "roc_auc")

# 7. Finalize workflow with best hyperparameters
final_rf_wf <- finalize_workflow(rf_wf, best_rf)

# 8. Fit to full training data
final_rf_model <- fit(final_rf_wf, data = train)

# 9. Save final model
save(final_rf_model, file = "final_randomforest_model.Rdata")