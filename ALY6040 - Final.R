library(readr)
library(ggplot2)
library(dplyr)
library(mice)

library(tidyverse)
library(lattice)
library(caret)
library(Amelia)
library(Rcpp)
library(GGally)
library(carData)
library(car)
library(Metrics)
library(rsample)
library(pROC)
library(Matrix)
library(glmnet)
library(rpart)
library(rpart.plot)
library(pROC)
library(ROCR)
library(kableExtra)

# import data
insurance <- read.csv("~/Desktop/6040/Final/insurance.csv")

head(insurance)
summary(insurance)
str(insurance)

# check null
md.pattern(insurance)

# Create Agegroup & data pre-process
insurance <- as_tibble(insurance)
levels(insurance$sex)<-c("F","M")
#Create Agegroup column
insurance %>% 
  select_if(is.numeric) %>% 
  map_dbl(~max(.x))
insurance <-insurance %>% 
  mutate(Agegroup=as.factor(findInterval(age,c(18,35,50,80))))
levels(insurance$Agegroup)<-c("Youth","Mid Aged","Old")
levels(insurance$smoker)<-c("N","Y")
levels(insurance$region)<-c("NE","NW","SE","SW")


# EDA

# binwith = 1, hist for age
ggplot(insurance, aes(x = age)) + geom_histogram(col = "black", fill = "cornflowerblue", binwidth = 1) +
  ggtitle("Histogram of Age") + xlab("age") + ylab("Count") +
  geom_text(stat = 'count', aes(label=..count..), colour = "black", vjust=-1, size=2.5)

# Individual medical costs billed by health insurance
ggplot(insurance, aes(x = charges)) + geom_histogram(bins = 50, col = "black", fill = "cornflowerblue") + 
  ggtitle("Histogram of Charges billed by health insurance ") + xlab("charges") + ylab("Count")
# almost no people above 50000
# right-skewed

# boxplot for Individual medical costs billed by health insurance
ggplot(insurance, aes(x = charges)) + geom_boxplot() + 
  ggtitle("Boxplot of Charges") + xlab("charges") + ylab("")

# Scatter plot of price and bmi, by smoker
ggplot(insurance, aes(x = bmi, y = charges, colour = smoker)) + geom_point() + 
  labs(colour= "smoker") + xlab("bmi") + ylab("charges") +
  ggtitle("Scatter plot of price and bmi , by smoker")

# hist of charges, fill = smoker
ggplot(insurance, aes(x = charges, fill = smoker)) + geom_histogram(bins = 50, col = "black") + 
  ggtitle("Histogram of Charges billed by health insurance ") + xlab("charges") + ylab("Count")

# Boxplot plot of price, by region
ggplot(insurance, aes(x = charges, colour = region)) + geom_boxplot() + 
  labs(colour= "region") + xlab("charges") +
  ggtitle("Scatter plot of price, by region") 

# Visualise distribution of charges by agegroup,sex and region
ggplot(insurance, aes(region,charges,fill=sex))+geom_boxplot()+facet_grid(~Agegroup)+
  ggtitle("Insurance Charge distribution by Age, Sex and region")+
  scale_fill_manual(values=c("whitesmoke","cornflowerblue"))

insurance %>% 
filter(charges>=30000) %>% 
  ggplot(aes(region,charges,fill=sex))+geom_boxplot()+facet_grid(~Agegroup)+
  ggtitle("Outlier Charges") +
  scale_fill_manual(values=c("whitesmoke","cornflowerblue"))

suppressMessages(print(
  insurance %>% 
    filter(charges>=30000) %>% 
    ggpairs(columns=c(1,6,7),aes(fill=sex))+
    labs(title="Correlation between age and charges for Outliers")+
    theme(plot.title=element_text(hjust=0.5,color="navy")) 
))

suppressMessages(print(
  insurance %>% 
    filter(charges>=30000) %>% 
    select_if(is.numeric) %>% 
    ggcorr(low = "magenta", mid = "goldenrod", high = "navy",label=T,
           label_color="whitesmoke")+
    labs(title="Correlation Plot for Numeric Data")+
    theme(plot.title=element_text(hjust=0.5,color="navy")) 
)
)

suppressMessages(print(
  insurance %>% 
    select_if(is.numeric) %>% 
    ggcorr(low = "magenta", mid = "goldenrod", high = "navy",label=T,
           label_color="gray4",palette="RdBu")+
    labs(title="Correlation Plot for Numeric Data")+
    theme(plot.title=element_text(hjust=0.5,color="navy")) 
)
)

print(insurance %>% 
            select(smoker,charges,age) %>% 
            ggpairs(aes(col=smoker)))

out <- insurance %>% 
  filter(charges>=40000)

head(out)

# linear regression
ins_model_null <- lm(charges ~ 1, data=insurance)
ins_model_full <- lm(charges ~ ., data=insurance)
step(ins_model_null,
     scope = list(upper=ins_model_full),
     direction="both",
     data= insurance)   

ins_model_final <- lm(formula = charges ~ smoker + age + bmi + children + Agegroup + 
     region, data = insurance)

summary(ins_model_final)

vif(ins_model_final)
insurance$predict = predict(ins_model_final)
plot(predict ~ charges,
     data= insurance,
     pch = 16,
     xlab="Actual response value",
     ylab="Predicted response value")
abline(0,1, col="blue", lwd=2)
hist(residuals(ins_model_final),
     col="darkgray")

#Divide the dataset into a training and validation set for some machine learning predictions
trainds <- createDataPartition(insurance$Agegroup,p=0.7,list=F)
validate <- insurance[-trainds,] 
trainds <- insurance[trainds,]  

#Set metric and control
control<-trainControl(method="cv",number=10)
metric<-"RMSE" 

#Set up models 

set.seed(123)
fit.knn<-train(charges~.,data=trainds,method="knn",trControl=control,metric=metric) 

set.seed(123)
fit.svm<-train(charges~.,data=trainds,method="svmRadial",trControl=control,metric=metric) 

set.seed(123)
fit.gbm<-train(charges~.,data=trainds,method="gbm",trControl=control,metric=metric,
               verbose=F) 

set.seed(123)
fit.xgb<-train(charges~.,data=trainds,method="xgbTree",trControl=control,metric=metric,
               verbose=F) 

set.seed(123) 
fit.rf<-train(charges~.,data=trainds,method="rf",trControl=control,metric=metric,
              verbose=F) 

results<-resamples(list(knn=fit.knn,svm=fit.svm,xgb=fit.xgb,gbm=fit.gbm,rf=fit.rf))

# Visualize model "Accuracies"
dotplot(results,main="Model Training Results")

# Gradient Boosting: Model Details
getTrainPerf(fit.gbm)

# XGBoost model details
getTrainPerf(fit.xgb)

# Support Vector Machine Model Details
getTrainPerf(fit.svm)

# RandomForest Model Details and Feature Importance
getTrainPerf(fit.rf)

plot(varImp(fit.rf),main="Model Feature Importance-Random Forest")

predicted<-predict(fit.gbm,validate)
plot(fit.gbm,main="GBM")

test_perf<-rmse(validate$charges,predicted) 
paste0("RSE is ",rse(validate$charges,predicted))
paste0("RMSE is ",test_perf)

# logistic & decision tree model

insurance2 <- read.csv("~/Desktop/6040/week 2/insurance.csv")
summary(insurance)

# split data 
insurance2$smoker = ifelse(insurance2$smoker == 'yes',1,0)

set.seed(123)
ins_split <- initial_split(insurance2, prop = .7)
ins_train <- training(ins_split)
ins_test  <- testing(ins_split)

# logistic regression

ins_log <- glm(smoker ~ ., 
                    data = ins_train, family = "binomial")

#Fit Test of Logistic Regression Model

prob<-predict(object = ins_log, newdata=ins_test, type = "response")
pred<-ifelse(prob >= 0.5, "yes", "no")
pred<-factor(pred,levels = c("no","yes"),order=TRUE)
f<-table(ins_test$smoker,pred)
f

# calculate ROC curve
pred_lm = predict(ins_log, type='response', newdata=ins_test)
rocr.pred.lr = prediction(predictions = pred_lm, labels = ins_test$smoker)
rocr.perf.lr = performance(rocr.pred.lr, measure = "tpr", x.measure = "fpr")
rocr.auc.lr = as.numeric(performance(rocr.pred.lr, "auc")@y.values)

# print ROC AUC
rocr.auc.lr

# plot ROC curve
plot(rocr.perf.lr,
     lwd = 3, colorize = TRUE,
     print.cutoffs.at = seq(0, 1, by = 0.1),
     text.adj = c(-0.2, 1.7),
     main = 'ROC Curve')
mtext(paste('Logistic Regression - auc : ', round(rocr.auc.lr, 5)))
abline(0, 1, col = "red", lty = 2)

# decision tree

ins_decision_tr = rpart(formula = smoker ~ .,
                         data = ins_train, method = "class")

rpart.plot(ins_decision_tr,type = 3, digits = 3, fallen.leaves = TRUE)

# calculate ROC curve
pred.DT = predict(ins_decision_tr, newdata = ins_test, type = 'prob')
rocr.pred = prediction(predictions = pred.DT[,2], labels = ins_test$smoker)
rocr.perf = performance(rocr.pred, measure = "tpr", x.measure = "fpr")
rocr.auc = as.numeric(performance(rocr.pred, "auc")@y.values)

# print ROC AUC
rocr.auc

# plot ROC curve
plot(rocr.perf,
     lwd = 3, colorize = TRUE,
     print.cutoffs.at = seq(0, 1, by = 0.1),
     text.adj = c(-0.2, 1.7),
     main = 'ROC Curve')
mtext(paste('Decision Tree - auc : ', round(rocr.auc, 5)))
abline(0, 1, col = "red", lty = 2)

# plot both ROC curves side by side for an easy comparison
par(mfrow=c(1,2))

# plot ROC curve for Decision Tree
plot(rocr.perf,
     lwd = 3, colorize = TRUE,
     print.cutoffs.at = seq(0, 1, by = 0.1),
     text.adj = c(-0.2, 1.7),
     main = 'ROC Curve')
mtext(paste('Decision Tree - auc : ', round(rocr.auc, 5)))
abline(0, 1, col = "red", lty = 2)

# plot ROC curve for Logistic Regression
plot(rocr.perf.lr,
     lwd = 3, colorize = TRUE,
     print.cutoffs.at = seq(0, 1, by = 0.1),
     text.adj = c(-0.2, 1.7),
     main = 'ROC Curve')
mtext(paste('Logistic Regression - auc : ', round(rocr.auc.lr, 5)))
abline(0, 1, col = "red", lty = 2)

