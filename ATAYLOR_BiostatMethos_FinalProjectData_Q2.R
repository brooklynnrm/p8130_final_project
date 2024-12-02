# Load Libraries
library('tidyr')
library('dplyr')
library('ggplot2')
library('survival')
library('survminer')
library('caret')
library('pROC')
library(MASS)


# Final Project - Problem 2
p2_data <- read_csv("Project_2_data.csv")
summary(p2_data)
colSums(is.na(p2_data))

# Categorical Variables to Factors
p2_data$Race <- as.factor(p2_data$Race)
p2_data$Marital.Status <- as.factor(p2_data$Marital.Status)
p2_data$T.Stage <- as.factor(p2_data$T.Stage)
p2_data$N.Stage <- as.factor(p2_data$N.Stage)
p2_data$X6th.Stage <- as.factor(p2_data$X6th.Stage)
p2_data$Differentiate <- as.factor(p2_data$Differentiate)
p2_data$Grade <- as.factor(p2_data$Grade)
p2_data$A.Stage <- as.factor(p2_data$A.Stage)
p2_data$Estrogen.Status <- as.factor(p2_data$Estrogen.Status)
p2_data$Progesterone.Status <- as.factor(p2_data$Progesterone.Status)
p2_data$Status <- as.factor(p2_data$Status) 

summary(p2_data)

# Brooklynn working with snake case
p2_data = breastcancer_df

# Check for missing values
colSums(is.na(p2_data))

# Pairwise relationship (for numerical variables)
pairs(p2_data[, c("age", "tumor_size", "regional_node_examined", "reginol_node_positive")])

# Visualize survival time
ggplot(p2_data, aes(x = survival_months, group = status)) + 
  geom_histogram(binwidth = 5, alpha = 0.7, position = "identity") +
  #scale_fill_manual(values = c(1 = "darkblue", 0 = "darkgreen")) +
  labs(title = "Distribution of Survival Months", x = "Months", y = "Frequency") +
    theme_minimal()

# Survival Analysis
# Kaplan-Meier survival curve 
km_fit <- survfit(Surv(survival_months, status == 1) ~ 1, data = p2_data)
ggsurvplot(km_fit, data = p2_data, conf.int = TRUE, risk.table = TRUE)
summary(km_fit)

#Kaplan-Meier by Race
km_fit_race <- survfit(Surv(survival_months, status == 1) ~ race, data = p2_data)
ggsurvplot(km_fit_race, data = p2_data, conf.int = FALSE, risk.table = TRUE)
summary(km_fit_race)

# Cox Proportional Hazards Model
# Fit a Cox proportional hazards model
cox_model <- coxph(Surv(survival_months, status == 1) ~ age + race + marital_status + 
                     t_stage + n_stage + sixth_stage + differentiate + grade +
                     a_stage + tumor_size + estrogen_status + 
                     progesterone_status + regional_node_examined + 
                     reginol_node_positive, data = p2_data)
summary(cox_model)

# Check proportional hazards assumptions
cox.zph(cox_model)

# View model violation: Visualize Schoenfeld Residuals to better understand the violation

# plot schoenfeld residuals for all predictors 
plot(cox.zph(cox_model))

# plot schoenfeld residuals for residuals, Estrogen Status 
plot(cox.zph(cox_model), var = "estrogen_status")

# plot schoenfeld residuals for Progesterone Status
plot(cox.zph(cox_model), var = "progesterone_status")

# Stratify the Cox Model by Estrogen and Progesterone Status
cox_model_strat <- coxph(Surv(survival_months, status == 1) ~ age + race + marital_status + 
                           t_stage + n_stage + sixth_stage + differentiate + grade +
                           a_stage + tumor_size + regional_node_examined + 
                           reginol_node_positive + strata(estrogen_status, progesterone_status), data = p2_data)
summary(cox_model_strat)

# check hazards assumption for strat data
cox.zph(cox_model_strat)

# Strat data plots
plot(cox.zph(cox_model_strat))


# Evaluate Model Performance
# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(p2_data$status, p = 0.7, list = FALSE)
train_data <- p2_data[train_index, ]
test_data <- p2_data[-train_index, ]

# Predict risk scores on test data
risk_scores <- predict(cox_model, newdata = test_data, type = "risk")

# ROC curve
roc_obj <- roc(test_data$status, risk_scores, levels = c(0:1))
plot(roc_obj, main = "ROC Curve")
auc(roc_obj)

# Fairness Evaluation by Race
# Stratify predictions by race
for(r in unique (p2_data$race)) {
  race_data <- test_data[test_data$race == r, ]
  race_risk_scores <- predict(cox_model, newdata = race_data, type = "risk")
  roc_race <- roc(race_data$status, race_risk_scores, levels = c(0:1))
  print(paste("AUC for race:", r))
  print (auc(roc_race))
  }

# Example approach: Recalibrate model using balanced sampling or adjusted loss functions
balanced_train <- train_data %>%
  group_by(race) %>%
  sample_n(size = min(table(train_data$race)))

balanced_cox_model <- coxph(Surv(survival_months, status == 0) ~ age + race + marital_status + 
                              t_stage + n_stage + sixth_stage + differentiate + grade +
                              a_stage + tumor_size + regional_node_examined + 
                              reginol_node_positive, data = balanced_train)

# Evaluate fairness again
for(r in unique(p2_data$race)) {
  race_data <- test_data[test_data$race == r, ]
  race_risk_scores <- predict(balanced_cox_model, newdata = race_data, type = "risk")
  roc_race <- roc(race_data$status, race_risk_scores, levels = c(0:1))
  print(paste("AUC for race:", r))
  print(auc(roc_race))
}

# --------- end
