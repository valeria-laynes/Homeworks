library(RWeka)
library(caret)
library("readxl")
library(e1071)



# TASK 4

## Question 1: C4.5
model <- read_excel('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\Question1.xlsx')
model_train <- model[1:6,4:7]
test <- model[7:12,4:7]

set.seed(1958)  # set a seed to get replicable results
train <- createFolds(model_train$Label, k=10)
C45Fit <- train(Label ~., method="J48", data=model_train,tuneLength = 5,trControl = trainControl(method="cv", indexOut=train))
print(C45Fit)
print(C45Fit$finalModel)
pred1 <- predict(C45Fit, newdata=test)
print(pred1)

## Question 2: C4.5
model2 <- read_excel('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\Question2.xlsx')
model_train2 <- model2[1:14,]
test2 <- model2[15:15,]

set.seed(1958)  # set a seed to get replicable results
train <- createFolds(model_train2$Play, k=10)
C45Fit <- train(Play ~., method="J48", data=model_train2,tuneLength = 5,trControl = trainControl(method="cv", indexOut=train))
print(C45Fit)
print(C45Fit$finalModel)
pred2 <- predict(C45Fit, newdata=test2)
print(pred2)




# TASK 5 - Question 2 - C4.5 Model
model5 <- read_excel('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\task5.xlsx')
model_train5 <- model5[1:24,]
test5 <- model5[25:36,]
actual5 <- test5[,'Label']

set.seed(1958)  # set a seed to get replicable results
train <- createFolds(model_train5$Label, k=10)
C45Fit <- train(Label ~., method="J48", data=model_train5,tuneLength = 5,trControl = trainControl(method="cv", indexOut=train))
print(C45Fit)
print(C45Fit$finalModel)
pred5 <- predict(C45Fit, newdata=test5)
print(pred5)
actual5 <- as.factor(actual5$Label)

cm <- confusionMatrix(pred5,actual5, positive = 'Win')
print(cm$byClass)
print(cm$overall)
