#Set the working directory and import the data set
setwd("C:/Users/cyu13/OneDrive/Desktop")
set.seed(101)
data <- read.csv('Admission_Predict_Ver1.1.csv')

#inspect the data set
str(data)
summary(data)
head(data)
any(is.na(data))

#install and use neuralnet
install.packages('neuralnet')
library(neuralnet)

#normalize the data set with max-min scale
maxs <- apply(data[,2:9],2,max)
mins <- apply(data[,2:9],2,min)
scaled.data <- as.data.frame(scale(data[,2:9],center=mins,scale=maxs-mins))

#inspect the scaled data set
head(scaled.data)

#install and use caTools to split and train the data set
install.packages('caTools')
library(caTools)
split=sample.split(scaled.data$Chance.of.Admit,SplitRatio =0.7)
train=subset(scaled.data,split==T)
test=subset(scaled.data,split==F)

#apply neuralnet package
n <- names(train)
f <- Chance.of.Admit ~ GRE.Score+TOEFL.Score+University.Rating+SOP+LOR+CGPA+Research
nn <- neuralnet(f,data=train,hidden=c(5,3),linear.output = T)

#plot the model to see the weights on each connection
plot(nn)

#compute predictions off test set
predicted.nn.values <- compute(nn,test[,1:7])

#It returns a list
str(predicted.nn.values)

# Convert back to non-scaled predictions
true.predictions <- predicted.nn.values$net.result*(max(data$Chance.of.Admit)-min(data$Chance.of.Admit))+min(data$Chance.of.Admit)


# Convert the test data
test.r <- (test$Chance.of.Admit)*(max(data$Chance.of.Admit)-min(data$Chance.of.Admit))+min(data$Chance.of.Admit)

# Check the Mean Squared Error
MSE.nn <- sum(test.r-true.predictions^2)/(nrow(test))

#visualize error
error.df <- data.frame(test.r, true.predictions)
head(error.df)
library(ggplot2)
ggplot(error.df,aes(x=test.r,y=true.predictions))+geom_point()+stat_smooth()


