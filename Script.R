# Title: "Forecasting the price of Italian Red Wine"
# Author: "Viktoriia Ilina"
# Date: "July 2021"

# Loading required packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(mltools)) install.packages("mltools", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(dplyr)
library(data.table)
library(readxl)
library(lattice)
library(ggplot2)
library(caTools)
library(caret)
library(e1071)
library(class)
library(xgboost)
library(mltools)

# Exploratory data analysis

# Important pre-step: download the file from GitHub to your working directory!

my_data <- read_excel("Red.xlsx")

head(my_data)

# Deleting irrelevant column

my_data <- my_data[-1]

# Transforming the data

my_data <- transform(
  my_data,
  Producer = as.factor(Producer),
  Grape = as.factor(Grape),
  Classification = as.factor(Classification),
  Alcohol = as.numeric(Alcohol),
  Year = as.factor(Year),
  Aging = as.factor(Aging),
  Price = as.numeric(Price)
)

# Checking the result

summary(my_data)

# Price distribution plot

histogram(my_data$Price, type = "count", 
          main = "Distribution of bottle prices", 
          xlab = "Price in euros",
          col = "gold2")

# Information about producers

Producers <- my_data %>% 
  group_by(Producer) %>%
  summarize(Count = n()) %>%
  arrange(desc(Count)) %>% 
  print()

sum(with(Producers, Count == 1))

# Distribution of grape variety

Grape_variety <- my_data %>% 
  group_by(Grape) %>%
  summarize(Count = n()) 

ggplot(data = Grape_variety, aes(x = reorder(Grape, Count), y = Count)) + geom_bar(stat = 'identity', aes(fill = Count)) + coord_flip() + theme_classic() + scale_fill_gradient(low = 'gold', high = 'gold3') + labs(title = 'Grape variety distribution', x = 'Variety of grape', y = 'Number of bottles')

# Alcohol content

sum(my_data$Alcohol >= 14)/nrow(my_data)

# Wines by year

Vintage <- my_data %>% 
  group_by(Year) %>%
  summarize(Count = n()) %>%
  arrange(desc(Count)) %>% 
  print()

# Plot of median wine price by grape harvest

ggplot(data = my_data) + geom_boxplot(mapping = aes(x = reorder(Year, Price, FUN = median), y = Price), fill = 'gold') + theme_classic() + labs(title = 'Median wine price by grape harvest', x = 'Vintage', y = 'Price')

# Plot of median wine price by kind of aging

ggplot(data = my_data) + geom_boxplot(mapping = aes(x = reorder(Aging, Price, FUN = median), y = Price), fill = 'gold') + theme_classic() + labs(title = 'Median wine price by kind of aging', x = 'Type of aging', y = 'Price')

# Methodology and analysis

# Creating categorical variable based on range and removing original data

my_data$Category <- cut(my_data$Price, c(0, 10, 25, 50, 75, 100, 200, 325))
levels(my_data$Category) <- c("<10", "10-25", "25-50", "50-75", "75-100", "100-200", ">200")
my_data$Price <- NULL

# For further analysis, convert "Alcohol" column content to factor

my_data$Alcohol <- as.factor(my_data$Alcohol)

# Checking the dataset

head(my_data)

# Splitting the data

set.seed(2) # set the seed to make the partion reproducible

sample <- sample.split(my_data$Category, SplitRatio = .8)
train <- subset(my_data, sample == TRUE)
test <- subset(my_data, sample == FALSE)

# Naive Bayes model

set.seed(120) # set the seed to make the prediction reproducible

# Fitting the model

model_naive <- naiveBayes(Category ~ ., data = train)

# Predicting on test data

prediction_naive <- predict(model_naive, newdata = test)

# Model evaluation

confusionMatrix(test$Category, prediction_naive)

# Support Vector Machines model

set.seed(2) # set the seed to make the prediction reproducible

# Training the model

model_svm <- svm(Category ~ ., data=train, 
                 method="C-classification", kernal="radial", 
                 gamma=0.1, cost=3)

# Predicting on test data

prediction_svm <- predict(model_svm, test)

# Model evaluation

confusionMatrix(test$Category, prediction_svm)

# eXtreme Gradient Boosting model

# XGBoost requires the classes to be in an integer format, starting with 0. 
# So, we have to convert "Category" factor to the proper format 

Categories <- my_data$Category
label <- as.integer(my_data$Category)-1
my_data$Category <- NULL

#Encoding all other variables to dummy variables

my_data <- one_hot(as.data.table(my_data))

# Splitting the data for training and testing

set.seed(2) # set the seed to make the partion reproducible

n <- nrow(my_data)
index <- sample(n, floor(0.8*n))
train <- as.matrix(my_data[index,])
train_label <- label[index]
test <- as.matrix(my_data[-index,])
test_label <- label[-index]

# Creating the xgb.DMatrix objects

xgb.train <- xgb.DMatrix(data = train,label = train_label)
xgb.test <- xgb.DMatrix(data = test,label = test_label)

# Define the parameters for multinomial classification

num_class <- length(levels(Categories))
xgb_params <- list(
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

set.seed(120) # set the seed to make the prediction reproducible

# Training the XGBoost model

xgb.fit <- xgb.train(
  params = xgb_params,
  data = xgb.train,
  nrounds = 100,
  early_stopping_rounds = 10,
  watchlist = list(val1 = xgb.train,val2 = xgb.test),
  verbose = 0
)

# Predicting on test data

xgb.pred <- predict(xgb.fit, test, reshape=T)
xgb.pred <- as.data.frame(xgb.pred)
colnames(xgb.pred) <- levels(Categories)

xgb.pred$prediction <- apply(xgb.pred, 1, function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label <- levels(Categories)[test_label+1]

# Model evaluation

confusionMatrix(factor(xgb.pred$prediction),
                factor(xgb.pred$label),
                mode = "everything")

