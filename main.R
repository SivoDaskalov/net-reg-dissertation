setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("batch_tools.R")

n = 100
transFactorsCount = 20
regulatedGenesPerTF = 10

batchResults = runBatch(n, transFactorsCount, regulatedGenesPerTF, methods = c("grace"))
# batchResults = runBatch(n, transFactorsCount, regulatedGenesPerTF)
unpackBatchResults(batchResults)
# save.image("data/N100P220.RData")

grace.lambda.1 = grace.lambda.2 = 10 * seq(from = 3, by = 1, length = 6)
grace.lambda.1 = grace.lambda.2 = 10 ^ seq(from = -2, by = 1, length = 6)
cvxGraceRes = convexGraceTuning(X = Xtu, Y = Ytu[,1], network = edges, degrees = degrees, lambda.1 = grace.lambda.1, lambda.2 = grace.lambda.2, k = 10)
optCvxGraceTuning = cvxGraceRes$results
matlabGraceTuning = oldCvxGrace$results
rGraceTuning = graceModels[[1]]$tuning
