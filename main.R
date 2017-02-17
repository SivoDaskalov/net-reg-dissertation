setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("batch_tools.R")

n = 100
transFactorsCount = 20
regulatedGenesPerTF = 10

batchResults = runBatch(n, transFactorsCount, regulatedGenesPerTF, methods = c("grace"))
# batchResults = runBatch(n, transFactorsCount, regulatedGenesPerTF)
unpackBatchResults(batchResults)
# save.image("data/N100P220.RData")

grace.lambda.1 = grace.lambda.2 = 10 ^ seq(from = -2, by = 1, length = 6)
graceFit = grace(xtr = Xtr, ytr = Ytr[,1], xtu = Xtu, ytu = Ytu[,1], network, degrees, 
                 lambda.1 = grace.lambda.1, lambda.2 = grace.lambda.2, 
                 adjustments = NULL, k = 10, norun = false)
cvxGracePred = predict(object = graceFit, data = Xts)
mean((cvxGracePred - Yts[,1])^2)
graceFit$coefficients
