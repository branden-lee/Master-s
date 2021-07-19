std.diff <- function(u,z,w) 
{ 
  # for variables other than unordered categorical variables compute mean differences 
  # mean(u[z==1]) gives the mean of u for the treatment group 
  # weighted.mean() is a function to calculate weighted mean 
  # u[z==0],w[z==0] select values of u and the weights for the comparison group 
  # weighted.mean(u[z==0],w[z==0],na.rm=TRUE): weighted mean for the comparison group 
  # sd(u[z==1], na.rm=T) calculates the standard deviation for the treatment group 
  
  if(!is.factor(u)) 
  { 
    sd1 <- sd(u[z==1], na.rm=T) 
    if(sd1 > 0) 
    { 
      result <- abs(mean(u[z==1],na.rm=TRUE)- 
                      weighted.mean(u[z==0],w[z==0],na.rm=TRUE))/sd1 
    } else 
    { 
      result <- 0 
      warning("Covariate with standard deviation 0.") 
    } 
  } 
  
  # for factors compute differences in percentages in each category 
  # for(u.level in levels(u) creates a loop that repeats for each level of  
  #  the categorical variable 
  # as.numeric(u==u.level) creates as 0-1 variable indicating u is equal to 
  #  u.level the current level of the for loop 
  # std.diff(as.numeric(u==u.level),z,w)) calculates the absolute  
  #   standardized difference of the indicator variable 
  else 
  { 
    result <- NULL 
    for(u.level in levels(u)) 
    { 
      result <- c(result, std.diff(as.numeric(u==u.level),z,w)) 
    } 
  } 
  return(result) 
} 
