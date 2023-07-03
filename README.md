# OkCupid: Predict Male / Female profiles with Naive Bayes

This is a machine learning project that predicts the gender of an OkCupid dating profile based on different profile prompt question responses 
using the Naive Bayes algorithm. The project includes a dataset of OkCupid dating profiles and their corresponding genders, which is used to train and test the algorithm.

What do we want to predict and why? 
- Our goal is to predict the gender of individuals based on the language used in their profile prompts. By analyzing these profiles and making accurate predictions, we can improve matchmaking algorithms and tailor marketing strategies to better reach each gender.

```
set.seed(196)
df = read.csv("data/okcupid_profiles.csv")
```

## Data Cleaning and Feature Engineering

To prepare our dataset for categorical analysis, we convert several columns into factors. Additionally, we extract the year from the “last online” column to assess the duration of user inactivity. Notably, all profiles were last online in either 2011 or 2012. This finding suggests that the data was either collected during that period or that all profiles have since become inactive. Furthermore, we observe that 91 out of 59,946 profile locations did not contain the keyword “California.” This suggests that the dataset only includes users from California, and that these 91 observations likely belong to out-of-state visitors or data errors.

```
set.seed(196)
df = group_locations_by_region(df)
df_clean = data_cleaning(df)
df_clean = feature_engineering(df_clean)
df_clean = df_clean[sample(nrow(df_clean)),]
```

## EDA Plots

- A histogram of orientation showing the frequency distribution of sexual orientation among the users. The vast majority of users identified as straight at 86.1%, while a small percentage identified as gay or bisexual at 9.3% and 4.6% respectively. There were only a few missing values in this column.
- A histogram of sex shows the frequency distribution of biological sex. In this dataset, there are slightly more male profiles (60%) than female profiles (40%). There were no profiles with missing sex data.
- A histogram of ages showing the frequency distribution across all age ranges in the dataset. The majority of users are between the ages of 18 and 29, with fewer users in older age ranges.
- A histogram of ages showing the frequency distribution for users between the ages of 18 and 29.
- A histogram of region location showing the frequency distribution of user locations across different regions in California. In this dataset, about 52% of users are located in San Francisco, 21% are in the East Bay Region, and 26% are in other regions in California.

```
# Plot histogram of orientation
plot_frequency(df, orientation)
```
![plot1](https://user-images.githubusercontent.com/94821306/232624251-b762e615-20c5-4aad-96eb-dd443bdb15e4.png)
```
# Plot histogram of sex
plot_frequency(df, sex)
```
![plot2](https://user-images.githubusercontent.com/94821306/232624330-48b8cff3-f5ad-4b32-876b-93dc03bf2d2f.png)
```
# Plot histogram of all ages
plot_age_histogram(df)
```
![plot3](https://user-images.githubusercontent.com/94821306/232624366-cc90ca02-ef6b-4a99-81d1-db6a9e3805b5.png)
```
# Plot histogram of ages 18-29
plot_age_histogram(df, age_range = c(18, 29))
```
![plot4](https://user-images.githubusercontent.com/94821306/232624423-3e1c6e64-f13b-47ac-8ed0-d1849f0c82b6.png)
```
# Plot histogram of region location
plot_frequency(df, region)
```
![plot5](https://user-images.githubusercontent.com/94821306/232624449-228289ab-23bd-4bc8-927b-264e88229ea6.png)

## Train / Test Split

We randomly sample 5% of the cleaned data to create a smaller dataset for training and testing our models. Additionally, we used an 80-20 split to partition the data into training and test sets, respectively. This allows us to evaluate the performance of our models on new data while minimizing overfitting to the training set.

```
set.seed(196)  # Set seed for reproducibility
# Create a stratified sample of 5% from the "sex_binary" column
stratified_sample = sample.split(df_clean$sex_binary, SplitRatio = 0.05)

# Split data into 80% training and 20% testing
split = sample.split(df_clean$sex_binary, SplitRatio = 0.8)
# Subset the clean data with the stratified sample
stratified_data = df_clean[stratified_sample, ]

# Split the stratified data into training and testing sets with an 80/20 ratio
split = sample.split(stratified_data$sex_binary, SplitRatio = 0.8)

# Create training and testing data frames
train_df = stratified_data[split, ]
test_df = stratified_data[!split, ]
```

## Train/Test Preprocessing

We then create a corpus from the set of tokens, perform stemming, and then convert the corpus to a document-term matrix for both the training and testing datasets. This is done to prepare the text data for use in the classification model.

```
# Combines essay prompts 0 - 9 into one for each profile
train_combined_essays = combine_essays(train_df)
test_combined_essays = combine_essays(test_df)
# Create corpus from the set of tokens and performs stemming
train_corpus = Corpus(VectorSource(train_combined_essays))
train_corpus = tm_map(train_corpus, stemDocument)
```

```
## Warning in tm_map.SimpleCorpus(train_corpus, stemDocument): transformation
## drops documents
```

```
test_corpus = Corpus(VectorSource(test_combined_essays))
test_corpus = tm_map(test_corpus, stemDocument)
```

```
## Warning in tm_map.SimpleCorpus(test_corpus, stemDocument): transformation drops
## documents
```

```
# Convert corpus to document-term matrix
train_dtm = DocumentTermMatrix(train_corpus)
train_matrix = as.matrix(train_dtm)
test_dtm = DocumentTermMatrix(test_corpus)
test_matrix = as.matrix(test_dtm)
```

## Train/Test Sparse Terms Removal

We apply frequency-based filtering to remove terms with document frequency less than a certain threshold. Specifically, we test different percentages and remove terms with document frequency less than each percentage. Then, we convert the sparse matrix to a dense matrix for each percentage hyperparameter, which will be used as input for the Naive Bayes classifier.

```
# Remove terms with document frequency less than 98%, 99%, and 99.9%
train_dtm_reduced_999 = removeSparseTerms(train_dtm, 0.999)
train_dtm_reduced_99 = removeSparseTerms(train_dtm, 0.99)
train_dtm_reduced_98 = removeSparseTerms(train_dtm, 0.98)
# Convert the sparse matrix to a dense matrix
train_matrix_reduced_999 = as.matrix(train_dtm_reduced_999)
train_matrix_reduced_99 = as.matrix(train_dtm_reduced_99)
train_matrix_reduced_98 = as.matrix(train_dtm_reduced_98)
# Remove terms with document frequency less than 98%, 99%, and 99.9%
test_dtm_reduced_999 = removeSparseTerms(test_dtm, 0.999)
test_dtm_reduced_99 = removeSparseTerms(test_dtm, 0.99)
test_dtm_reduced_98 = removeSparseTerms(test_dtm, 0.98)
# Convert the sparse matrix to a dense matrix
test_matrix_reduced_999 = as.matrix(test_dtm_reduced_999)
test_matrix_reduced_99 = as.matrix(test_dtm_reduced_99)
test_matrix_reduced_98 = as.matrix(test_dtm_reduced_98)
```

## Training / Hyperparameter Tuning / Performance Evaluation

We will train a Naive Bayes classifier on a training matrix to predict gender from a transformed binary column. To optimize the model, we will tune the Laplace smoothing hyperparameter for each of the sparse term parameters. We will evaluate the performance of each parameter by making predictions on the test matrix and creating confusion matrices for further analysis.

Note: The following cell may take some time to run, and it may output several warnings or error messages. However, it is necessary for the analysis as it trains the candidate models for later evaluation.

```
set.seed(196)
  # your R code here
  laplace_values = c(0,0.1,0.5,1,5,10)
  nb_models_98 = train_nb_classifiers(train_matrix_reduced_98, train_df$sex_binary, laplace_values)
nb_models_99 = train_nb_classifiers(train_matrix_reduced_99, train_df$sex_binary, laplace_values)
nb_models_999 = train_nb_classifiers(train_matrix_reduced_999, train_df$sex_binary, laplace_values)
# Get predictions on the test set for each model
predictions_list_98 = list()
predictions_list_99 = list()
predictions_list_999 = list()
for (i in 1:length(nb_models_98)) {
  predictions = predict(nb_models_98[[i]], test_matrix_reduced_98)
  predictions_list_98[[i]] = predictions
}
```

```
# Create the confusion matrix
confusion_matrices_98 = list()
confusion_matrices_99 = list()
confusion_matrices_999 = list()

for (i in 1:length(predictions_list_98)) {
  confusion_matrices_98[[i]] = table(actual = test_df$sex_binary, predicted = predictions_list_98[[i]])
}
for (i in 1:length(predictions_list_99)) {
  confusion_matrices_99[[i]] = table(actual = test_df$sex_binary, predicted = predictions_list_99[[i]])
}
for (i in 1:length(predictions_list_999)) {
  confusion_matrices_999[[i]] = table(actual = test_df$sex_binary, predicted = predictions_list_999[[i]])
}
```

## Calculating Metrics

We calculate the average performance of the Naive Bayes classifier using five different metrics that are evaluated from the information present in each confusion matrix.

```
# Define a function to calculate accuracy, sensitivity, and specificity
calc_metrics = function(cm) {
  total = sum(cm)
  accuracy = sum(diag(cm)) / total
  sensitivity = cm[2, 2] / sum(cm[2, ])
  specificity = cm[1, 1] / sum(cm[1, ])
  precision = cm[2, 2] / sum(cm[, 2])
  f1_score = 2 * precision * sensitivity / (precision + sensitivity)
  return(list(accuracy = accuracy, sensitivity = sensitivity, specificity = specificity, precision = precision, f1_score = f1_score))

}
# Calculate metrics for each set of predictions
metrics_list_98 = lapply(confusion_matrices_98, calc_metrics)
metrics_list_99 = lapply(confusion_matrices_99, calc_metrics)
metrics_list_999 = lapply(confusion_matrices_999, calc_metrics)

# Calculate the average performance metrics across all folds
avg_metrics_98 = lapply(metrics_list_98, function(x) sapply(x, mean))
avg_metrics_99 = lapply(metrics_list_99, function(x) sapply(x, mean))
avg_metrics_999 = lapply(metrics_list_999, function(x) sapply(x, mean))
```

## Metrics Plot Preprocessing

After the metrics are calculated for each combination of hyperparameters, a full data frame is constructed with the average metric values for each combination. This data frame makes it easy to compare the performance of different models and to create visualizations for further analysis.

```
# Combine average performance metrics into a data frame
metrics_df = data.frame(
  Accuracy = unlist(lapply(c(avg_metrics_98, avg_metrics_99, avg_metrics_999), function(x) x[1])),
  Sensitivity = unlist(lapply(c(avg_metrics_98, avg_metrics_99, avg_metrics_999), function(x) x[2])),
  Specificity = unlist(lapply(c(avg_metrics_98, avg_metrics_99, avg_metrics_999), function(x) x[3])),
  Precision = unlist(lapply(c(avg_metrics_98, avg_metrics_99, avg_metrics_999), function(x) x[4])),
  F1_Score = unlist(lapply(c(avg_metrics_98, avg_metrics_99, avg_metrics_999), function(x) x[5])),
  Laplace = rep(c(0, 0.1, 0.5, 1, 5, 10), times = 3),
  Features = rep(c("98%", "99%", "99.9%"), each = 6)
)
```

## Metrics Plots

Based on the analysis, we find that increasing the percentage of variance explained by the PCA model from 98% to 99% results in improved performance across all five evaluation metrics, including accuracy, sensitivity, specificity, precision, and F1 score. We also observe that varying the Laplace smoothing parameter had little effect on model performance.

However, further increasing the percentage of variance explained beyond 99% resulted in a decrease in performance for all metrics except for precision, which remained at 100%. The F1 score also decreased to 0%, indicating that the model was unable to achieve a balance between precision and recall at this level of variance explained.

Therefore, the PCA model with 99% variance is recommended as it achieves the best overall performance across all metrics.

```
# Plot all metrics against feature percentage
ggplot(metrics_df, aes(x = Features)) +
  geom_line(aes(y = Accuracy, color = "Accuracy", group = 1), linewidth = 1) +
  geom_point(aes(y = Accuracy, color = "Accuracy", group = 1), size = 3) +
  geom_line(aes(y = Sensitivity, color = "Sensitivity", group = 1), linewidth = 1) +
  geom_point(aes(y = Sensitivity, color = "Sensitivity", group = 1), size = 3) +
  geom_line(aes(y = Specificity, color = "Specificity", group = 1), linewidth = 1) +
  geom_point(aes(y = Specificity, color = "Specificity", group = 1), size = 3) +
  geom_line(aes(y = Precision, color = "Precision", group = 1), linewidth = 1) +
  geom_point(aes(y = Precision, color = "Precision", group = 1), size = 3) +
  geom_line(aes(y = F1_Score, color = "F1 Score", group = 1), linewidth = 1) +
  geom_point(aes(y = F1_Score, color = "F1 Score", group = 1), size = 3) +
  labs(title = "Performance Metrics vs. Feature Percentage",
       x = "Feature Percentage",
       y = "Metric Value",
       color = "Metric") +
  scale_color_manual(values = c("Accuracy" = "red", "Sensitivity" = "blue",
                                "Specificity" = "green", "Precision" = "purple",
                                "F1 Score" = "orange"),
                     labels = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score")) +
  theme_dark() +
  theme(plot.background = element_rect(fill = "lightgray"))
```
![plot6](https://user-images.githubusercontent.com/94821306/232624492-5074ea48-c2b2-4c4e-85e2-aa6478c8712d.png)

## SUMMARY 
In this project, we investigated the effectiveness of Naive Bayes classifier in predicting the sex of individuals based on their OkCupid dating profiles using a dataset of anonymized profiles from OkCupid in the United States. Due to limited processing power and the large size of the data (~130MB), we opted for an 80/20 train-test split for model evaluation instead of cross-validation, which is more computationally expensive. Although a simple train-test split may result in higher variance in the estimated performance metrics compared to cross-validation, we believe it is a reasonable trade-off between computational efficiency and model performance given our resources.

SOURCE: [https://www.kaggle.com/datasets/yashsrivastava51213/okcupid-profiles-dataset](https://www.kaggle.com/datasets/yashsrivastava51213/okcupid-profiles-dataset)


