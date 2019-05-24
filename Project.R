load("C:/Users/HP/Desktop/Semester 2/Statistical Machine Learning/Project/backpain.RData")
data<-dat
set.seed(18201354)
# Check if the categorical variables are factors. If not, make them as factors.
str(data)

data$SurityRating<-as.factor(data$SurityRating)
# Converting the response variable as factors
data$PainDiagnosis<-as.factor(data$PainDiagnosis)

res<-matrix(NA,100,8) # Matrix to store the result

library(adabag)
library(nnet)
library(rpart)
library(randomForest)
library(kernlab)
# Start simulation to look at this
iterlim <- 100
for (iter in 1:iterlim)
{  
  # Bootstrap the data
  # Use the bootstrap sample as training data
  # The remaining data is used for validation and test 
  N<-nrow(data)
  train_data<-sample(1:N,replace=TRUE)
  train_data<-sort(train_data)
  remaining_data<-setdiff(1:N,train_data)
  r<-length(remaining_data)
  valid_data<-sample(setdiff(1:N,train_data),size=0.5*r)
  valid_data<-sort(valid_data)
  test_data<-setdiff(1:N,union(train_data,valid_data))

#----------------------------------------------------------#
#                    LOGISTIC REGRESSION                   #
#----------------------------------------------------------#
  fit.l<-multinom(PainDiagnosis~.,data=data,subset=train_data)
  summary(fit.l)
  pred.l<-predict(fit.l,type="class",newdata=data[valid_data,])
  tab.l<-table(data$PainDiagnosis[valid_data],pred.l)
  acc.l<-sum(diag(tab.l))/sum(tab.l)
  res[iter,1]<-acc.l 

#----------------------------------------------------------#
#                    CLASSIFICATION TREE                   #
#----------------------------------------------------------#
  fit.c<-rpart(PainDiagnosis~.,data=data,subset=train_data)
  summary(fit.c)
  pred.c<-predict(fit.c,type="class",newdata=data[valid_data,])
  tab.c<-table(data$PainDiagnosis[valid_data],pred.c)
  acc.c<-sum(diag(tab.c))/sum(tab.c)
  res[iter,2]<-acc.c

#----------------------------------------------------------#
#                         BAGGING                          #
#----------------------------------------------------------#
  fitbag<-bagging(PainDiagnosis~.,data=data[train_data,],mfinal=25)
  pred.b<-predict(fitbag,type="class",newdata=data[valid_data,])
  tab.b<-pred.b$confusion
  acc.b<-(sum(tab.b)-sum(diag(tab.b)))/sum(tab.b)
  res[iter,3]<-acc.b

#----------------------------------------------------------#
#                       RANDOM FOREST                      #
#----------------------------------------------------------#
  fit.rf<-randomForest(PainDiagnosis~.,data=data[train_data,])
  pred.rf<-predict(fit.rf,type="class",newdata=data[valid_data,])
  tab.rf<-table(data$PainDiagnosis[valid_data],pred.rf)
  acc.rf<-sum(diag(tab.rf))/sum(tab.rf)
  res[iter,4]<-acc.rf

#----------------------------------------------------------#
#                         BOOSTING                         #
#----------------------------------------------------------#
  fit.boost<-boosting(PainDiagnosis~.,data=data[train_data,],mfinal=25)
  pred.boost<-predict.boosting(fit.boost,data[valid_data,])
  acc.boost<-1-pred.boost$error
  res[iter,5]<-acc.boost

#----------------------------------------------------------#
#                  SUPPORT VECTOR MACHINE                  #
#----------------------------------------------------------#
  fit.svm<-ksvm(PainDiagnosis~.,data=data[train_data,])
  pred.svm <- predict(fit.svm,data[valid_data,])
  tab.svm <- table(data$PainDiagnosis[valid_data],pred.svm)
  acc.svm<-sum(diag(tab.svm))/sum(tab.svm)
  res[iter,6]<-acc.svm

  max.acc<-which.max(res[iter,1:6])

  if(max.acc=="1")
  {
    pred.l<-predict(fit.l,type="class",newdata=data[test_data,])
    tab <- table(data$PainDiagnosis[test_data],pred.l)
    acc <- sum(diag(tab))/sum(tab)
    res[iter,7] <- "Logistic"
    res[iter,8] <- acc
  }
  else if(max.acc=="2")
  {
    pred.c<-predict(fit.c,type="class",newdata=data[test_data,])
    tab <- table(data$PainDiagnosis[test_data],pred.c)
    acc <- sum(diag(tab))/sum(tab)
    res[iter,7] <- "Classification Tree"
    res[iter,8] <- acc
  }
  else if(max.acc=="3")
  {
    pred.b<-predict(fitbag,type="class",newdata=data[test_data,])
    tab<-pred.b$confusion
    acc<-(sum(tab.b)-sum(diag(tab.b)))/sum(tab.b)
    res[iter,7] <- "Bagging"
    res[iter,8] <- acc
  }
  else if(max.acc=="4")
  {
    pred.rf<-predict(fit.rf,type="class",newdata=data[test_data,])
    tab<-table(data$PainDiagnosis[test_data],pred.rf)
    acc<-sum(diag(tab.rf))/sum(tab.rf)
    res[iter,7] <- "Random Forest"
    res[iter,8] <- acc
  }
  else if(max.acc=="5")
  {
    pred.boost<-predict.boosting(fit.boost,data[test_data,])
    acc<-1-pred.boost$error
    res[iter,7] <- "Boosting"
    res[iter,8] <- acc
  }
  else
  {
    pred.svm <- predict(fit.svm,data[test_data,])
    tab <- table(data$PainDiagnosis[test_data],pred.svm)
    acc<-sum(diag(tab.svm))/sum(tab.svm)
    res[iter,7] <- "SVM"
    res[iter,8] <- acc
  }
}

colnames(res)<-c("Validation Logistic","Validation Classification Tree","Validation Bagging","Validation Random Forest","Validation Boosting","Validation SVM","Best Method","Test Accuracy using best method")

result<-as.data.frame(apply(res[,-7],2, as.numeric))
summary(result)
 
tabl<-table(res[,7])
tabl

best<-as.numeric(res[res[,7]=="Random Forest",8]) # Accuracy of best model on test data
summary(best)

# Variable Importance
val<-varImp(fit.rf)    # Values
varImpPlot(fit.rf) # Plot
apply(val,2,sort)