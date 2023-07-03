# MAIN FEATURE ENGINEERING METHOD
feature_engineering = function(df) {
  df = extract_last_online_year(df)
  df = transform_to_factor(df)
  df = transform_sex_binary(df)
  return(df)
}

# Method to extract the year from the last_online column
extract_last_online_year = function(df) {
  last_online_year = year(df[, "last_online"])
  df[, "last_online_year"] = last_online_year
  return(df)
}
# Method to transform status and last_online_year columns to factor
transform_to_factor = function(df) {
  df$status = as.factor(df$status)
  df$last_online_year = as.factor(df$last_online_year)
  return(df)
}
# Method to transform sex column to binary in new column
transform_sex_binary = function(df) {
  df$sex_binary = ifelse(df$sex == "m", 0, 1)
  return(df)
}

# Method to create a new column with the region for each location
group_locations_by_region = function(df) {
  df = df %>%
    mutate(region = case_when(
      grepl("los angeles|ventura|orange|san diego", location, ignore.case = TRUE) ~ "Southern California",
      grepl("san francisco", location, ignore.case = TRUE) ~ "San Francisco",
      grepl("san jose", location, ignore.case = TRUE) ~ "San Jose",
      grepl("oakland|berkeley|palo alto|stanford", location, ignore.case = TRUE) ~ "East Bay",
      grepl("peninsula", location, ignore.case = TRUE) ~ "Peninsula",
      grepl("sacramento|davis|stockton", location, ignore.case = TRUE) ~ "Central Valley",
      grepl("santa barbara|san luis obispo|monterey", location, ignore.case = TRUE) ~ "Central Coast",
      TRUE ~ "Other"
    ))
  return(df)
}

# Combines all essays from a single profile into one
combine_essays = function(df) {
  essay_cols = grep("^essay\\d+_tokens$", colnames(df), value = TRUE)
  combined_essays = vector("character", nrow(df))
  combined_sex = vector("character", nrow(df))  # New line
  for (i in seq_len(nrow(df))) {
    essay_tokens = unlist(df[i, essay_cols])
    combined_essays[i] = paste(essay_tokens, collapse = " ")
    combined_sex[i] = df[i, "sex_binary"]  # New line
  }
  combined_data = data.frame(essays = combined_essays, sex = combined_sex)  # New line
  return(combined_data)  # Modified line
}

