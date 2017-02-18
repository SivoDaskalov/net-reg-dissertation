setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("batch_tools.R")

n = 100
transFactorsCount = 200
regulatedGenesPerTF = 10

# batchResults = runBatch(n, transFactorsCount, regulatedGenesPerTF, methods = c("agrace"))
batchResults = runBatch(n, transFactorsCount, regulatedGenesPerTF)
unpackBatchResults(batchResults)
save.image("data/ConvexN100P2200.RData")

# grace.lambda.1 = grace.lambda.2 = 10 ^ seq(from = -2, by = 1, length = 6)
# graceFit = grace(xtr = Xtr, ytr = Ytr[,1], xtu = Xtu, ytu = Ytu[,1], network, degrees, 
#                  lambda.1 = grace.lambda.1, lambda.2 = grace.lambda.2, 
#                  adjustments = NULL, k = 10, norun = false)
# cvxGracePred = predict(object = graceFit, data = Xts)
# evalBetaStatistics(trueBeta = as.numeric(betas[1,]), betaHat = as.numeric(unlist(graceFit$coefficients)), trueBetaIndices = 1:44)
# evalErrors(actual = Yts[,1], predicted = cvxGracePred)