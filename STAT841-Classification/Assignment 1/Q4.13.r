library(logistf)

### part a
df <- read.csv("C:/Users/brand/OneDrive/Desktop/hunger_games.csv", header=TRUE)

model <- glm(surv_day1 ~ female + age + volunteer + has_name, family=binomial,
             data=df)
sink("4.13a.txt")
summary(model)


### part b
model2 <- logistf(surv_day1 ~ female + age + volunteer + has_name, family=binomial,
             data=df)
sink("4.13b.txt")
summary(model2)