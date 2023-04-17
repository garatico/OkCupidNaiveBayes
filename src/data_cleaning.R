# MAIN DATA CLEANING METHOD
data_cleaning = function(df) {
  df = generate_uuid(df)
  df = filter_california_locations(df)
  df = filter_empty_essays(df)
  df = filter_straight(df)
  df = remove_unused_columns(df, c("body_type", "diet", "drinks", "drugs", "education", "ethinicity", "height", "income", "job", "offspring", "pets", "religion", "sign", "smokes"))
  df = remove_stopwords_and_tokenize(df)
  return(df)
}

# Method to generate unique ID for each row using the UUID package
generate_uuid = function(df) {
  df$ID = UUIDgenerate(n = nrow(df))
  return(df)
}
# Keep only entries with California locations
filter_california_locations = function(df) {
  df = df[grepl("california", df$location, ignore.case = TRUE), ]
  return(df)
}
# Method to filter out rows with empty essay columns
filter_empty_essays = function(df) {
  df = df %>%
    filter(rowSums(df[, paste0("essay", 0:9)] == "") < 10)
  return(df)
}
# Keep only straight orientation
filter_straight = function(df) {
  filtered_df = df %>% filter(sex %in% c("m", "f"), orientation == "straight")
  return(filtered_df)
}
# Function to remove unused columns
remove_unused_columns = function(df, columns_to_remove) {
  df[, !(colnames(df) %in% columns_to_remove)]
}
# Removes stop words and tokenizes each essay column
remove_stopwords_and_tokenize = function(df) {
  tokens_df <- df %>% select(ID)
  for (i in 0:9) {
    colname <- paste0("essay", i)
    
    # remove punctuation
    no_punct_colname <- paste0(colname, "_no_punct")
    df <- df %>% mutate(!!no_punct_colname := stringr::str_remove_all(!!sym(colname), "[^[:alnum:] ]"))
    
    # tokenize text and remove stop words
    tokens_colname <- paste0(colname, "_tokens")
    df_tokens <- df %>%
      select(ID, !!no_punct_colname) %>%
      unnest_tokens(word, !!no_punct_colname) %>%
      anti_join(stop_words, by = "word") %>%
      group_by(ID) %>%
      summarise(!!tokens_colname := paste(word, collapse = " ")) %>%
      ungroup()
    # merge tokens with tokens_df
    tokens_df <- inner_join(tokens_df, df_tokens, by = "ID")
  }
  df_tokens <- inner_join(tokens_df, df %>% select(-matches("essay[0-9]|_no_punct")), by = "ID")
  return(df_tokens)
}


# Keeps only entries with San Francisco as location
filter_san_francisco = function(df) {
  df_san_francisco = df %>%
    filter(grepl("san francisco", location, ignore.case = TRUE))
  return(df_san_francisco)
}
# Keep only entries within age range
filter_age <- function(df, min_age, max_age) {
  df_filtered <- df %>%
    filter(age >= min_age & age <= max_age)
  return(df_filtered)
}


# Get the term frequency of each word
term_freq = function(tdm, chunk_size = 10000) {
  num_rows = nrow(tdm)           # Get number of rows in TDM
  term_freq = data.frame(word = rownames(tdm), freq = 0)   # Empty data frame to store

  # Loop over chunks of tdm and sum term frequencies
  for (i in seq(1, num_rows, chunk_size)) {
    chunk = tdm[i:min(i + chunk_size - 1, num_rows), ]
    term_freq[i:min(i + chunk_size - 1, num_rows), "freq"] = rowSums(as.matrix(chunk))
  }
  return(term_freq)
}







# NORMALIZATION?
normalization <- function(df) {
  
}

# PARTITION TASKS?
partitioning <- function(df) {
  
}




