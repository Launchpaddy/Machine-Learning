#review <- read.csv("~/machine/review1mil.csv")
review_random_wtext <- review[sample(nrow(review), 1000),  c(3,5,7,8,9)]
review_limit <- review_random_wtext[, c(4,5)]

install.packages('dplyr')
install.packages('tm')
install.packages('caret')

library(dplyr)
library(tm)
library(caret)


binary_reviews <- review_limit %>% mutate(useful_new = if_else(useful >= 1, 1, 0))
binary_training_id <- sample.int(nrow(binary_reviews), size = nrow(binary_reviews)*0.8)
binary_training <- binary_reviews[binary_training_id,]
binary_testing <- binary_reviews[-binary_training_id,]


corpus_toy <- Corpus(VectorSource(binary_training$text))
tdm_toy <- DocumentTermMatrix(corpus_toy, list(removePunctuation = TRUE, 
                                               removeNumbers = TRUE))

training_set_toy <- as.matrix(tdm_toy)

training_set_toy <- cbind(training_set_toy, binary_training$useful_new)

colnames(training_set_toy)[ncol(training_set_toy)] <- "y"

training_set_toy <- as.data.frame(training_set_toy)
training_set_toy$y <- as.factor(training_set_toy$y)


review_toy_model <- train(y ~., data = training_set_toy, method = 'svmLinear3')
