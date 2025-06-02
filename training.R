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
library(vip)
library(ggplot2)

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

# Define recipe for preprocessing
em_recipe <- recipe(emig ~ q4113+ 
                      q409 +
                      q1002+
                      q1005_combined +
                      q4116+
                      q103a2+
                      q102b+
                      q106+
                      q101+
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
                      q1001c+
                      q514+
                      q5181+
                      q2011+
                      q513_combined+
                      q1010b+
                      q2013+
                      q7013+
                      q4062+
                      q2042+
                      q20422+
                      q261b2+
                      q2044+
                      q2043, data = train) %>%
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
  print(n = 100)

###################################################
## LOGISTIC REGRESSION

logistic_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Combine into a workflow
logistic_wf <- workflow() %>%
  add_model(logistic_spec) %>%
  add_recipe(em_recipe)

# Fit model with cross-validation
set.seed(42)
cv_results <- fit_resamples(
  logistic_wf,
  resamples = folds,
  metrics = metric_set(accuracy, roc_auc, sensitivity, specificity),
  control = control_resamples(save_pred = TRUE)
)

# Show top results
cv_results %>%
  show_best(metric = "roc_auc", n = 10)

# Select best model (if there were tuning parameters)
best_model <- select_best(cv_results, metric = "roc_auc")

# finalize and fit to full training data
final_logistic_wf <- finalize_workflow(logistic_wf, best_model)

final_logistic_reg <- fit(final_logistic_wf, data = train)

## RANDOM FORESTS
# Set up parallel backend
registerDoFuture()
plan(multisession, workers = parallel::detectCores() - 1)

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
set.seed(42)
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
m <- fit(final_rf_wf, data = train)

# Save final model
save(m, file = "final_model.Rdata")

cv_results_rf %>%
  show_best(metric = "roc_auc", n = 1)

collect_metrics(cv_results_rf)

# --------- variable importance plot ------------------

rf_fit <- extract_fit_parsnip(m)$fit

rename_lookup <- c(
  "age" = "Age",
  "q1001c" = "Years living in current area",
  "q4113_No" = "Facebook use: No",
  "q409_I.do.not.use.the.internet" = "Internet use: No",
  "q1005_combined_Housewife" = "Employment: Housewife",
  "q4116_No" = "Instagram Use: No",
  "q1002_Female" = "Gender: Female",
  "q1005_combined_Unemployed" = "Employment: Unemployed",
  "q101_Very.bad" = "Economic situation: Very bad",
  "q1010_Married" = "Marital Status: Married"
)

top_vars <- vip::vi(rf_fit) %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 10)

top_vars <- top_vars %>%
  mutate(
    PrettyName = if_else(
      !is.na(rename_lookup[Variable]),
      rename_lookup[Variable],
      Variable  # fallback to original
    )
  )

ggplot(top_vars, aes(x = reorder(PrettyName, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Variables by Importance (Random Forest)",
       x = "Variable",
       y = "Importance") +
  theme_minimal()




