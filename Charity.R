## Final Group Project
## PREDICT 422
## Padmini Vijay, Austin Harrison, Kevin Wong

### LOAD DATA
charity <- read.csv(file.choose())

### EXPLORATORY DATA ANALYSIS
str(charity)
summary(charity)
anyNA(charity[,-(22:23)]) # check for NAs in only predictor variables

## Examine distribution of variables
# histograms of of continous predictors
for( p in c("damt","avhv","incm","inca","plow","npro","tgif","lgif","rgif","tdon","tlag","agif")){
    hist(charity[[p]],main = paste("Histogram of" , p),xlab = p)
}

# boxplots of donor amount by categorical variables
for( p in c("home","chld","hinc","genf","wrat")){
    boxplot(damt~get(p),charity,main = paste("BoxPlot of damt vs" , p))
}

# frequency table of donors vs categorical variables
for( p in c("home","chld","hinc","genf","wrat")){
    print(p)
    with(charity,print(table(donr,get(p))))
}

# scatter plot of continuous variables
pairs(damt~avhv+incm+inca+plow,data=charity[charity$part=="train",])
pairs(damt~tdon+tlag+agif+npro+tgif+lgif,data=charity[charity$part=="train",])

# correlation
cor(charity[charity$part=="train",c(11:21,23)])

# predictor transformations using log
charity.t <- charity
for( p in c("avhv","incm","inca","tgif","lgif","rgif","agif")){
    charity.t[[p]] <- log(charity.t[[p]])
    par(mfrow=c(1,2))
    hist(charity[[p]],main = paste("Histogram of " , p),xlab = p)
    hist(charity.t[[p]],main = paste("Histogram of log" , p),xlab = p)
    par(mfrow=c(1,1))
}

## Set up data for analysis
data.train <- charity.t[charity$part=="train",]
x.train <- data.train[,2:21]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[,2:21]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,2:21]

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)

data.train.std.c$donr = as.factor(data.train.std.c$donr)
data.valid.std.c$donr = as.factor(data.valid.std.c$donr)


### CLASSIFICATION MODELING

## MODEL 1 - Neural network
library(nnet)

model.nnet1 <- nnet(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + 
                        I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + 
                        tgif + lgif + rgif + tdon + tlag + agif, data.train.std.c, 
                        size = 2)
post.valid.nnet1 <- predict(model.nnet1, data.valid.std.c)


# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.nnet1 <- cumsum(14.5*c.valid[order(post.valid.nnet1, decreasing = T)] - 2)
plot(profit.nnet1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.nnet1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.nnet1)) # report number of mailings and maximum profit
# 1311.0 11631.5

cutoff.nnet1 <- sort(post.valid.nnet1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.nnet1 <- ifelse(post.valid.nnet1 > cutoff.nnet1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.nnet1, c.valid) # classification table


## MODEL 2 - Neural network #2 with size adjustment to 3 hidden layers
model.nnet2 <- nnet(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + 
                        I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + 
                        tgif + lgif + rgif + tdon + tlag + agif, data.train.std.c, 
                    size = 3)
post.valid.nnet2 <- predict(model.nnet2, data.valid.std.c)

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.nnet2 <- cumsum(14.5*c.valid[order(post.valid.nnet2, decreasing = T)] - 2)
plot(profit.nnet2) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.nnet2) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.nnet2)) # report number of mailings and maximum profit
# 1267.0 11661.5

cutoff.nnet2 <- sort(post.valid.nnet2, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.nnet2 <- ifelse(post.valid.nnet2 > cutoff.nnet2, 1, 0) # mail to everyone above the cutoff
table(chat.valid.nnet2, c.valid) # classification table


## MODEL 3 - Gradient boosting model with parameter tuning using CV
library(caret)
fitControl <- trainControl(method="repeatedcv", number=5, repeats=5)
set.seed(1)
gbmFit <- train(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + 
                    hinc + I(hinc^2) + genf + wrat + avhv + incm + 
                    inca + plow + npro + tgif + lgif + rgif + tdon + 
                    tlag + agif, data = data.train.std.c,
                    method = "gbm", trControl = fitControl, verbose = FALSE)
gbmFit # optimal parameters seem to be shrinkage = 0.1, n.minobsinnode = 10, n.trees = 150, interaction.depth = 3

set.seed(1)
boost.model = gbm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + 
                      hinc + I(hinc^2) + genf + wrat + avhv + incm + 
                      inca + plow + npro + tgif + lgif + rgif + tdon + 
                      tlag + agif, data = data.train, distribution = "bernoulli", 
                      shrinkage = 0.1, n.minobsinnode = 10, n.trees = 150, interaction.depth = 3)
#chld, hinc, reg2 and home are most importart variables
summary(boost.model)

set.seed(1)
boost.prob.model = predict.gbm(boost.model, newdata = data.valid, n.trees = 150, type = "response")
boost.pred.model = rep("0", 2018)
boost.pred.model[boost.prob.model> .5] = "1"
table(boost.pred.model , c.valid) # correct prediction 900+947/2081
boost.err.model <- mean(boost.pred.model != c.valid)
boost.err.model # [1] 0.08473736

profit.boost <- cumsum(14.5*c.valid[order(boost.prob.model , decreasing=T)]-2)
# see how profits change as more mailings are made
plot(profit.boost, main = "Maximum Profit - Boosting") 
# number of mailings that maximizes profits
n.mail.boost <- which.max(profit.boost) 
# report number of mailings and maximum profit
c(n.mail.boost, max(profit.boost)) # 1236 11955.5

#cutoffs
cutoff.boost <- sort(boost.prob.model, decreasing=T)[n.mail.boost+1] # set cutoff based on number of mailings for max profit
chat.boost <- ifelse(boost.prob.model  > cutoff.boost, 1, 0) # mail to everyone above the cutoff
table(chat.boost, c.valid) # classification table
#778+995/2018
boost.donors = 241+995 # 1236
boost.profit = 14.5*995-2*1236 # 11955.5



### PREDICTION MODELING

## Neural network #1

nnet1 <- nnet(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + 
                        I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + 
                        tgif + lgif + rgif + tdon + tlag + agif, data = data.train.std.y, 
                        size = 2, linout = TRUE)
pred.nnet1 <- predict(nnet1, newdata = data.valid.std.y)

nnet1.mse <- mean((y.valid - pred.nnet1)^2)
nnet1.mse # [1] 1.884441
nnet1.se <- sd((y.valid - pred.nnet1)^2)/sqrt(n.valid.y)
nnet1.se # [1] 0.1882455


## Neural network #2
nnet2 <- nnet(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + 
                        I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + 
                        tgif + lgif + rgif + tdon + tlag + agif, data = data.train.std.y, 
                        size = 3, linout = TRUE)
pred.nnet2 <- predict(nnet2, newdata = data.valid.std.y)

nnet2.mse <- mean((y.valid - pred.nnet2)^2)
nnet2.mse # [1] 1.476413
nnet2.se <- sd((y.valid - pred.nnet2)^2)/sqrt(n.valid.y)
nnet2.se # [1] 0.1605676


## Gradient boosting model
set.seed(1)
gbm.model <- gbm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + 
                     hinc + I(hinc^2) + genf + wrat + avhv + incm + 
                     inca + plow + npro + tgif + lgif + rgif + tdon + 
                     tlag + agif, data = data.train.std.y, distribution = "gaussian", 
                     shrinkage = 0.1, n.minobsinnode = 10, n.trees = 150, interaction.depth = 3)
summary(gbm.model) #rgif, lgif, agif and reg4 are important variables
set.seed(1)
pred.gbm <- predict.gbm(gbm.model, newdata = data.valid.std.y, type = "response", n.trees = 150)

gbm.mse <- mean((y.valid - pred.gbm)^2)
gbm.mse # [1] 1.382862
gbm.se <- sd((y.valid - pred.gbm)^2)/sqrt(n.valid.y)
gbm.se # [1] 0.1600592