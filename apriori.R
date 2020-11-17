#install.packages('arules');
library(arules);
data(Groceries);

inspect(Groceries[1:5])

image(sample(Groceries, 100))

itemFrequencyPlot(Groceries, topN = 20)


groceryrules  <- apriori(Groceries, parameter = list(support = 0.006, confidence = .25, minlen = 2))

groceryrules1 <- apriori(Groceries, parameter = list(support = 0.01, confidence = 0, minlen = 2))
groceryrules2 <- apriori(Groceries, parameter = list(support = 0.001, confidence = .6, minlen = 2))

groceryrules3 <- apriori(Groceries, parameter = list(support = 0.000, confidence = .00, minlen = 2))
#summary(groceryrules)


inspect(groceryrules1)


inspect(sort(groceryrules1, by = "support") [1:5])
inspect(sort(groceryrules2, by = "confidence")[1:5])
inspect(sort(groceryrules3, by = "lift")[1:5])


inspect(sort(groceryrules, by = "support")[1:5])
inspect(sort(groceryrules, by = "confidence")[1:5])

inspect(sort(groceryrules, by = "lift")[1:5])

inspect(sort(groceryrules, by = "count")[1:5])


assign("myrulse", NULL, envir = .GlobalEnv)
myrules <- subset(groceryrules, rhs %in% "sausage")

inspect(myrules)
inspect(sort(myrules, by = "lift")[1:2])

