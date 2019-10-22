rm(list=ls())          
library(data.table)
train=fread("C:\\Users\\AJIT\\Documents\\amexpert\\train.csv",stringsAsFactors = T)
data=subset(train,select = -c(session_id,DateTime,user_id,campaign_id,webpage_id))
data=data[sample(nrow(data),5000), ]

sum(is.na(data))
library(caret)
preProcValues <- preProcess(data, method = c("knnImpute","center","scale"))
library('RANN')
train_processed <- predict(preProcValues, data)
sum(is.na(train_processed))
dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))
train_transformed$is_click<-as.factor(train_transformed$is_click)
id=sample(2,nrow(train_transformed),prob=c(.8,.3),replace=TRUE)
train=train_transformed[id==1,]
test=train_transformed[id==2,]
############################################################################################
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'is_click'
predictors<-names(newtrain)[!names(newtrain) %in% outcomeName]
bestf <- rfe(newtrain[,predictors], newtrain[,outcomeName],
             rfeControl = control)
bestf
#################################################################################################
y.dep <- 19
x.indep <- c(1:18)
#x.indep=c("product_category_1","product_category_2","age_level","product.J","product.C",
#  "user_group_id","product.D","product.B")
library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()
train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)
###############################GBM#############################################################
gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o,
                     ntrees = 10000, max_depth = 4, learn_rate = 0.01, seed = 1122)

predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
print(h2o.auc(h2o.performance(gbm.model, newdata = test.h2o)))

##############################################################################################
hyper_params = list(
  max_depth = seq(minDepth,maxDepth,1),
  sample_rate = seq(0.2,1,0.01),
  col_sample_rate = seq(0.2,1,0.01),
  col_sample_rate_per_tree = seq(0.2,1,0.01),
  col_sample_rate_change_per_level = seq(0.9,1.1,0.01),
  min_rows = 2^seq(0,log2(nrow(train.h2o))-1,1),
  nbins = 2^seq(4,10,1),
  nbins_cats = 2^seq(4,12,1),
  min_split_improvement = c(0,1e-8,1e-6,1e-4),
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")
)
search_criteria = list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 3600,
  max_models = 100,
  seed = 1234,
  stopping_rounds = 5,
  stopping_metric = "AUC",
  stopping_tolerance = 1e-3
)
grid <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "gbm",
  grid_id = "final_grid",
  x=x.indep,y=y.dep,
  training_frame = train.h2o,
  ntrees = 10000,
  learn_rate = 0.05,
  learn_rate_annealing = 0.99,
  max_runtime_secs = 3600,
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC",
  score_tree_interval = 10,
  seed = 1234
)
sortedGrid <- h2o.getGrid("final_grid", sort_by = "auc", decreasing = TRUE)
sortedGrid
gbm <- h2o.getModel(sortedGrid@model_ids[[1]])

for (i in 1:5) {
  gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
  cvgbm <- do.call(h2o.gbm,
                   ## update parameters in place
                   {
                     p <- gbm@parameters
                     p$model_id = NULL          ## do not overwrite the original grid model
                     p$training_frame =train.h2o      ## use the full dataset
                     p$validation_frame = NULL  ## no validation frame
                     p$nfolds = 5              ## cross-validation
                     p
                     #p$number_of_trees=250
                     #p$max_depth=10
                   }
  )
  print(gbm@model_id)
  print(cvgbm@model$cross_validation_metrics_summary[5,]) ## Pick out the "AUC" row
}
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
submit <- data.frame(session_id = test$session_id, is_click = predict.gbm$predict)
submit$is_click=ifelse(submit$is_click=='4.23080956110385',1,0)
write.csv(submit, file = "C:\\Users\\AJIT\\Documents\\amexpert\\s9.csv", row.names = FALSE)

##########################################DL###################################################
dlearning.model <- h2o.deeplearning(y = y.dep,
                                    x = x.indep,
                                    training_frame = train.h2o,
                                    epoch = 60,
                                    hidden = c(100,100),
                                    activation = "Rectifier",
                                    seed = 1122)
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))
submit <- data.frame(session_id = test$session_id, is_click = predict.dl2$predict)
submit$is_click=ifelse(submit$is_click=='4.23080956110385',1,0)
write.csv(submit, file = "C:\\Users\\AJIT\\Documents\\amexpert\\s5.csv", row.names = FALSE)

###########################################Sampling########################################################
#SMOTE
#over sampling
library(ROSE)
data_balanced_over <- ovun.sample(is_click ~ ., data = newtrain, method = "over",N = 863920)$data
table(data_balanced_over$class)
train.h2o=as.h2o(data_balanced_over)
#Under Sampling
data_balanced_under <- ovun.sample(class ~ ., data =train, method = "under", N = 62662, seed = 1)$data
table(data_balanced_under$class)
data.rose <- ROSE(class ~ ., data = train, seed = 1)$data
table(data.rose$class)
gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o,
                     ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1122)
h2o.varImp(object=gbm.model)

predict.gbm <- as.data.frame(h2o.predict(gb, test.h2o))
submit <- data.frame(session_id = test$session_id, is_click = predict.gbm$predict)
submit$is_click=ifelse(submit$is_click=='4.23080956110385',1,0)
write.csv(submit, file = "C:\\Users\\AJIT\\Documents\\amexpert\\s7.csv", row.names = FALSE)

###############################################################################################
rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, 
                                  training_frame = train.h2o, ntrees = 1000, mtries = 4, max_depth = 4, seed = 1122)
predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o))
h2o.varimp(rforest.model)
submit <- data.frame(session_id = test$session_id, is_click = predict.rforest$predict)
submit$is_click=ifelse(submit$is_click=='4.23080956110385',1,0)
write.csv(submit, file = "C:\\Users\\AJIT\\Documents\\amexpert\\s8.csv", row.names = FALSE)
