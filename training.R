rm(list=ls())

############################################################
# Template for final project                               #
############################################################

# Set your working directory here. 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load necessary packages here. All packages must
# be available for install via install.packages()
# or remotes::install_github().

library(pak)
pak::pkg_install("xgboost@1.7.11.1")

library(ROCR)
library(dplyr)
library(tidymodels)
library(xgboost)
library(doFuture)

# define function to assess classification performance
auc <- function(phat,y){
	pred <- ROCR::prediction(phat, y)
	perf <- ROCR::performance(pred,"auc")
	auc <- perf@y.values[[1]]
	return(auc)
}

# Load training data
load("AB4x_train.Rdata")

# Make sure that the original dataset is not altered
train_raw <- train  
train <- train_raw

#################################
### Understand data structure ###

# missing variables: how many and where
na_counts <- colSums(is.na(train))
na_vars <- na_counts[na_counts > 0]
na_vars # Missing values are in age and gender.
# Instead of removing, we impute missing values in the recipes

# Any non-numeric vars?
# Get the class of each column
var_classes <- sapply(train, class)
print(var_classes)
var_classes <- sapply(var_classes, `[`, 1)
table(var_classes)

# Make character variable "emig" to factor
train$emig <- factor(train$emig, levels = c("No", "Yes"))

###########################
### Data pre-processing ###

## Version 1: Likert scales (≥5 levels) to numeric
train_num <- train

# Define the names of the factor variables you want to convert
numeric_vars <- c("edu_combined", "income_categories", "q101a",
                  "q102", "q103a1", "q103a2", "q103a3", "q103a4",
                  "q103a5", "q103a6", "q261a1", "q261a2", "q261b1",
                  "q261b2", "q4061", "q4062", "q409", "q605",
                  "q6101", "q6106", "q7011", "q7012", "q7013",
                  "q7014", "q701b")

# Convert specified factor variables to numeric (1 for first level, 2 for second, etc.)
train_num <- train_num %>%
  mutate(across(all_of(numeric_vars), ~ as.numeric(factor(.x, levels = levels(.x)))))

# Make character variable to factor
train_num$emig <- as.factor(train_num$emig)

# Define recipe for preprocessing
em_recipe_num <- recipe(emig ~ q4113 + q409 +
                          q1005_combined +
                          q1002 +
                          q101+
                          q106+
                          q102b+
                          q4116+
                          q1+
                          q1010b+
                          q105+
                          q103a2+
                          q2044+
                          q2+
                          q20421+
                          q1074+
                          q2181+
                          q2011+
                          age+
                          q20422+
                          q4062+
                          q204a2+
                          q701b+
                          q2013+
                          edu_combined+
                          region+
                          q514+
                          q2042+
                          q2016+
                          q1001c+
                          q5181+
                          q261b2+
                          q103a2+
                          q2043+
                          q513_combined+
                          q1010+
                          q7013, data = train_num) %>%
  step_indicate_na(q1002) %>% # New gender/age columns to see if NAs
  step_indicate_na(age) %>%   # are predictive of emigration
  step_impute_mode(q1002) %>%  # Mode as only 5% NAs
  step_impute_median(age) %>%  # Median, as its robust to outliers
  step_dummy(all_nominal_predictors()) %>% # Only unordered factors one-hot-encoded
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# Create cross-validation setup
set.seed(42)
folds_num <- vfold_cv(train_num, v = 5, strata = emig)


## Version 1.2: Likert scales (≥4 levels) to numeric
train_num2 <- train

# Define the names of the factor variables you want to convert
numeric_vars2 <- c( "edu_combined", "income_categories", "q101",
                    "q101a", "q102", "q102b", "q103a1",
                    "q103a2", "q103a3", "q103a4", "q103a5", "q103a6",
                    "q105", "q106", "q2011", "q2012", "q2013", "q2014",
                    "q2016", "q20113", "q20120", "q202",  "q204a1",
                    "q204a2", "q210", "q216", "q2181", "q2182",
                    "q2185", "q261a1", "q261a2", "q261b1", "q261b2",
                    "q4061", "q4062", "q409",
                    "q514", "q5181", "q5182", "q5183",
                    "q5184", "q5185", "q5186",  "q523", "q6011",
                    "q6012", "q6013", "q6014", "q6018", "q60118",
                    "q6041", "q6043",  "q605", "q6062", "q6063",
                    "q6064", "q6071", "q6073", "q6074", "q6076", 
                    "q6077", "q6101", "q6106", "q7011", "q7012",
                    "q7013", "q7014", "q701b"
)

#"edu_combined", "income_categories",
#"q261a1", "q261a2", "q261b1", "q261b2",
#"q4061", "q4062", "q409",
#"q6101", "q6106",
#"q7011", "q7012", "q7013", "q7014", "q701b"

# Convert specified factor variables to numeric (1 for first level, 2 for second, etc.)
train_num2 <- train_num2 %>%
  mutate(across(all_of(numeric_vars2), ~ as.numeric(factor(.x, levels = levels(.x)))))

# Make character variable to factor
train_num2$emig <- as.factor(train_num2$emig)

# Define recipe for preprocessing
em_recipe_num2 <- recipe(emig ~ q4113 + q409 +
                           q1005_combined +
                           q1002 +
                           q101+
                           q106+
                           q102b+
                           q4116+
                           q1+
                           q1010b+
                           q105+
                           q103a2+
                           q2044+
                           q2+
                           q20421+
                           q1074+
                           q2181+
                           q2011+
                           age+
                           q20422+
                           q4062+
                           q204a2+
                           q701b+
                           q2013+
                           edu_combined+
                           region+
                           q514+
                           q2042+
                           q2016+
                           q1001c+
                           q5181+
                           q261b2+
                           q103a2+
                           q2043+
                           q513_combined+
                           q1010+
                           q7013, data = train_num2) %>%
  step_indicate_na(q1002) %>% # New gender/age columns to see if NAs
  step_indicate_na(age) %>%   # are predictive of emigration
  step_impute_mode(q1002) %>%  # Mode as only 5% NAs
  step_impute_median(age) %>%  # Median, as its robust to outliers
  step_dummy(all_nominal_predictors()) %>% # Only unordered factors one-hot-encoded
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# Create cross-validation setup
set.seed(42)
folds_num2 <- vfold_cv(train_num2, v = 5, strata = emig)


## Version 2: Likert scale variables (≥ 5 levels) as factors

# Make character variable to factor
train$emig <- as.factor(train$emig)

# Test differences train_num and train for Elastic net
# XGBoost: can handle factors

# Define recipe for preprocessing
em_recipe <- recipe(emig ~ q4113 + q409 +
                      q1005_combined +
                      q1002 +
                      q101+
                      q106+
                      q102b+
                      q4116+
                      edu_combined+
                      q701b+
                      q1010+
                      q2181+
                      q2016+
                      q1074+
                      region+
                      age+
                      q20421+
                      q105+
                      q103a2+
                      q1010b+
                      q103a2+
                      q2044+
                      q2011+
                      q20422+
                      q4062+
                      q204a2+
                      q701b+
                      q2013+
                      q514+
                      q2042+
                      q1001c+
                      q5181+
                      q261b2+
                      q103a2+
                      q2043+
                      q513_combined+
                      q7013, data = train) %>%
  step_indicate_na(q1002) %>% # New gender/age columns to see if NAs
  step_indicate_na(age) %>%   # are predictive of emigration
  step_impute_mode(q1002) %>%  # Mode as only 5% NAs
  step_impute_median(age) %>%  # Median, as its robust to outliers
  step_dummy(all_nominal_predictors()) %>% # Only unordered factors one-hot-encoded
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# Create cross-validation setup
set.seed(42)
folds <- vfold_cv(train, v = 5, strata = emig)

###################
### Build model ###

## PENALIZED LOG. CLASSIFICATION WITH ELASTIC NET
# Set up parallel backend
registerDoFuture()
plan(multisession, workers = parallel::detectCores() - 1)

penalized_spec <- logistic_reg(
  penalty = tune(), 
  mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

penalized_wf <- workflow() %>%
  add_model(penalized_spec) %>%
  add_recipe(em_recipe)

penalty_grid <- grid_regular(
  penalty(range = c(-4, 0), trans = log10_trans()),  # e.g., 10^-4 to 10^0
  mixture(range = c(0, 1)),  # 0 = ridge, 1 = lasso
  levels = 5
)

penalized_res <- tune_grid(
  penalized_wf,
  resamples = folds,
  grid = penalty_grid,
  metrics = metric_set(roc_auc, pr_auc, accuracy)
)

# Show best-performing model
penalized_res %>% 
  show_best(metric = "roc_auc", n = 10)

best_model <- select_best(penalized_res, metric = "roc_auc")

final_wf <- finalize_workflow(penalized_wf, best_model)

final_fit <- fit(final_wf, data = train)

# Which predictors were kept?
tidy(final_fit) %>%
  filter(term != "(Intercept)") %>%   # Remove intercept for clarity
  arrange(desc(abs(estimate))) %>%    # Sort by magnitude of effect
  print(n = 115)

#################################
## BOOSTING WITH XGBOOST
# Set up parallel backend
registerDoFuture()
plan(multisession, workers = parallel::detectCores() - 1)

xgb_spec <- boost_tree(
  trees = tune(),          # Number of boosting rounds (trees)
  tree_depth = tune(),     # Maximum depth of each tree
  learn_rate = tune(),      # Learning rate (shrinkage)
  loss_reduction = tune(),  # gamma
) %>%
  set_engine("xgboost",
             penalty = tune(),   # L2 regularization
             mixture = tune()     # L1 regularization
  ) %>%
  set_mode("classification")

# Workflow setup
xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_formula(emig ~ q4113 + q409 + q1005_combined + q1002 + q101 +
                q106 + q102b + q4116)

xgb_grid <- grid_regular(
  trees(range = c(100, 1000)),
  tree_depth(range = c(2, 8)),
  learn_rate(range = c(0.01, 0.3)),
  loss_reduction(range = c(0, 5)),      # gamma
  penalty(range = c(0, 5)),             # L2 regularization (lambda)
  mixture(range = c(0, 1)),             # L1 regularization (alpha)
  levels = 2
)

# Run the grid search with parallelization
xgb_tuned <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid = xgb_grid,
  metrics = metric_set(roc_auc, pr_auc),
  control = control_grid(save_pred = TRUE)
)

# Show best-performing parameters
xgb_tuned %>% 
  show_best(metric = "roc_auc", n = 5)

###############################
# Evaluate model
best_xgb <- select_best(xgb_tuned, metric = "roc_auc")

final_xgb_wf <- finalize_workflow(xgb_wf, best_xgb)

xgb_final_fit <- fit(final_xgb_wf, data = train)

# Extract feature importance
xgb_final <- extract_fit_parsnip(xgb_final_fit)$fit

xgb.importance(model = xgb_final) %>% 
  as_tibble() %>% 
  arrange(desc(Gain)) %>%
  print(n = 100)

# Simple plot
xgb.importance(model = xgb_final) %>%
  xgb.plot.importance(top_n = 100, measure = "Gain")

###################################################

# ~~~ ANNINA EXPLORATORY ANALYSIS ~~~

# ~~~ Logistic Regression ~~~
#tidymodels is imported above

logistic_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# 3. Combine into a workflow
logistic_wf <- workflow() %>%
  add_model(logistic_spec) %>%
  add_recipe(em_recipe)

# 4. Fit model with cross-validation
set.seed(123)
cv_results <- fit_resamples(
  logistic_wf,
  resamples = folds,
  metrics = metric_set(accuracy, roc_auc, sensitivity, specificity),
  control = control_resamples(save_pred = TRUE)
)

# 5. Show top results
cv_results %>%
  show_best(metric = "roc_auc", n = 10)

# 6. Select best model (if there were tuning parameters)
best_model <- select_best(cv_results, metric = "roc_auc")

# 7. Finalize and fit to full training data
final_logistic_wf <- finalize_workflow(logistic_wf, best_model)

final_logistic_reg <- fit(final_logistic_wf, data = train)

# 8. Save final model
save(final_logistic_reg, file = "final_logistic_model.Rdata")

# 9. Output performance metrics
collect_metrics(cv_results)

# ~~~ Random Forest ~~~

rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# Workflow using your recipe
rf_wf <- workflow() %>%
  add_recipe(em_recipe) %>%
  add_model(rf_spec)

# Create tuning grid
rf_grid <- grid_regular(
  mtry(range = c(2, 10)),
  min_n(range = c(2, 10)),
  levels = 5
)

# Cross-validation
set.seed(123)
cv_results_rf <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc, accuracy),
  control = control_grid(save_pred = TRUE)
)

# Select best hyperparameters
best_rf <- select_best(cv_results_rf, metric = "roc_auc")

# Finalize workflow
final_rf_wf <- finalize_workflow(rf_wf, best_rf)

# Fit final model
final_rf_model <- fit(final_rf_wf, data = train)

# Save final model
save(final_rf_model, file = "final_randomforest_model.Rdata")

cv_results_rf %>%
  show_best(metric = "roc_auc", n = 1)

collect_metrics(cv_results_rf)
