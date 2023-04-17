# Train a single Naive Bayes model using incremental updating
train_nb_classifier = function(dtm_train, dv, laplace_param) {
  # Calculate the number of rows in the training data and the frequency of progress updates
  n_rows = nrow(dtm_train)
  print_every = round(n_rows/10)
  # Convert the sparse matrix to a dense matrix as required by the naiveBayes function
  dtm_matrix_df = as.data.frame(as.matrix(dtm_train))
  
  # Train the initial model using the first row of the data
  nb_model = naiveBayes(x = dtm_matrix_df, y = dv, laplace = laplace_param)
  return(nb_model)
}

# Train multiple Naive Bayes models using different passed laplace parameters
train_nb_classifiers = function(dtm_train, dv, laplace_params) {
  # Initialize an empty list to store the models
  nb_models_list = list()
  
  # Loop through all laplace parameters and train a model for each
  for (laplace_param in laplace_params) {
    # Train the model
    nb_model = train_nb_classifier(dtm_train, dv, laplace_param)
    
    # Add the model to the list
    nb_models_list[[as.character(laplace_param)]] = nb_model
  }
  
  # Return the list of models
  return(nb_models_list)
}


#train_nb_models = function(df, label, laplace_params) {
#  results <- list()
#  for (i in 1:length(laplace_params)) {
#    nb_model <- naiveBayes(data, label, laplace = laplace_params[i])
#    nb_pred <- predict(nb_model, data)
#    nb_acc <- sum(nb_pred == label) / length(label)
#    results[[i]] <- list(model = nb_model, accuracy = nb_acc)
#  }
#  return(results)
#}

# Test Laplace = 1,2,3,4,5,6,7,8,9
#trained_nb_models = train_nb_models(data = df_clean, label = df_clean$sex_binary, laplace_params = c(0, 1, 2, 3, 4))



# Train the Naive Bayes classifier and print progress
#n_rows <- nrow(dtm_train_reduced)
#print_every <- round(n_rows/10)
#dtm_train_matrix_df <- as.data.frame(as.matrix(dtm_train_reduced))

#for (i in 1:n_rows) {
#  nb_model <- naiveBayes(x = dtm_train_matrix_df[i,], y = train_df$sex_binary[i], laplace = 1)
#  if (i %% print_every == 0) {
#    cat(sprintf("Trained on %d rows out of %d\n", i, n_rows))
#  }
#}

