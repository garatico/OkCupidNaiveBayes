plot_frequency <- function(df, column) {
  column <- enquo(column)
  ggplot(data = df, aes(x = !!column, fill = !!column)) +
    geom_bar(stat = "count") + 
    labs(x = quo_name(column), y = "Frequency", title = paste("Frequency of", quo_name(column))) +
    geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)))), 
              stat='count', 
              vjust=-0.5, size=3)
}

plot_sex_freq <- function(df) {
  ggplot(data = df, aes(x = sex, fill = sex)) +
    geom_bar(stat = "count") +
    labs(x = "Sex", y = "Frequency", title = "Frequency of Sex")
}

# Function to create histogram of ages
plot_age_histogram <- function(df, age_range = NULL) {
  # Filter data based on age_range
  if (!is.null(age_range)) {
    df <- df[df$age >= age_range[1] & df$age <= age_range[2], ]
  }
  
  ggplot(data = df, aes(x = age, fill = factor(age))) +
    geom_histogram(binwidth = 1) +
    labs(x = "Age", y = "Frequency", title = "Frequency of Age")
}






