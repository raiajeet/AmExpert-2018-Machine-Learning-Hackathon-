rm(list=ls())          
library(data.table)
train=fread("C:\\Users\\AJIT\\Documents\\amexpert\\train.csv",stringsAsFactors = T)
test=fread("C:\\Users\\AJIT\\Documents\\amexpert\\test.csv",stringsAsFactors = T)
test$is_click<-0
combi <- rbind(train, test)
data=subset(combi,select = -c(session_id,DateTime,user_id,campaign_id,webpage_id))
sum(is.na(data))
library(caret)
preProcValues <- preProcess(data, method = c("medianImpute","center","scale"))
library('RANN')
train_processed <- predict(preProcValues, data)
sum(is.na(train_processed))
dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))
train_transformed$is_click<-as.factor(train_transformed$is_click)
newtrain <- train_transformed[1:nrow(train),]
newtest <- train_transformed[-(1:nrow(train)),]
#################################################################################################
y.dep <- 19
x.indep <- c(1:18)
library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()
train.h2o <- as.h2o(newtrain)
test.h2o <- as.h2o(newtest)
###############################GBM#############################################################
gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o,
                     ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1122)

hyper_params = list( max_depth = c(4,6,8,12,16,20))
grid <- h2o.grid( hyper_params = hyper_params,search_criteria = list(strategy = "Cartesian"),
         algorithm="gbm",grid_id="depth_grid",y=y.dep, x=x.indep, training_frame = train.h2o,
         ntrees = 20000,  learn_rate = 0.05,learn_rate_annealing = 0.99,sample_rate = 0.8,
         col_sample_rate = 0.8,  seed = 1234,stopping_rounds = 5,
                  stopping_tolerance = 1e-4,
                  stopping_metric = "AUC",  score_tree_interval = 10)
sortedGrid <- h2o.getGrid("depth_grid", sort_by="auc", decreasing = TRUE)
topDepths = sortedGrid@summary_table$max_depth[1:5]
minDepth = min(as.numeric(topDepths))
maxDepth = max(as.numeric(topDepths))
sortedGrid

gbm <- h2o.getModel(sortedGrid@model_ids[[16]])

predict.gbm <- as.data.frame(h2o.predict(gbm, test.h2o))
submit <- data.frame(session_id = test$session_id, is_click = predict.gbm$predict)
submit$is_click=ifelse(submit$is_click=='4.23080956110385',1,0)
write.csv(submit, file = "C:\\Users\\AJIT\\Documents\\amexpert\\s8.csv", row.names = FALSE)
