### Boosting

set.seed(1)
model.gbm1 <- gbm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + 
                      hinc + I(hinc^2) + genf + wrat + avhv + incm + 
                      inca + plow + npro + tgif + lgif + rgif + tdon + 
                      tlag + agif, data.train,
                  distribution = "bernoulli", shrinkage = 0.1, 
                  n.minobsinnode = 10, n.trees = 150, interaction.depth = 3)

post.valid.gbm1 <- predict(model.gbm1, newdata = data.valid, 
                           n.trees = 150, type = "response")



profit.gbm1 <- cumsum(14.5*c.valid[order(post.valid.gbm1, decreasing=T)]-2)
plot(profit.gbm1, main = "Maximum Profit - GBM") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.gbm1) 
c(n.mail.valid, max(profit.gbm1))  
# 1236.0 11955.5

cutoff.gbm1 <- sort(post.valid.gbm1, decreasing=T)[n.mail.valid+1]  
chat.valid.gbm1 <- ifelse(post.valid.gbm1>cutoff.gbm1, 1, 0)  
table(chat.valid.gbm1, c.valid) # classification table


### Neural Network #1 with size=1
set.seed(1)
model.nnet1 <- nnet(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + 
                        I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + 
                        tgif + lgif + rgif + tdon + tlag + agif, data.train.std.c, 
                    size = 1)
post.valid.nnet1 <- predict(model.nnet1, data.valid.std.c)

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.nnet1 <- cumsum(14.5*c.valid[order(post.valid.nnet1, decreasing = T)] - 2)
plot(profit.nnet1, main = "Maximum Profit - NN #1") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.nnet1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.nnet1)) # report number of mailings and maximum profit
# 1045.0 10800.5

cutoff.nnet1 <- sort(post.valid.nnet1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.nnet1 <- ifelse(post.valid.nnet1 > cutoff.nnet1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.nnet1, c.valid) # classification table



### Neural Network #2 with size=10
set.seed(1)
model.nnet2 <- nnet(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + 
                        I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + 
                        tgif + lgif + rgif + tdon + tlag + agif, data.train.std.c, 
                    size = 10)
post.valid.nnet2 <- predict(model.nnet2, data.valid.std.c)

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.nnet2 <- cumsum(14.5*c.valid[order(post.valid.nnet2, decreasing = T)] - 2)
plot(profit.nnet2, main = "Maximum Profit - NN #2") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.nnet2) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.nnet2)) # report number of mailings and maximum profit
# 1289 11545

cutoff.nnet2 <- sort(post.valid.nnet2, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.nnet2 <- ifelse(post.valid.nnet2 > cutoff.nnet2, 1, 0) # mail to everyone above the cutoff
table(chat.valid.nnet2, c.valid) # classification table