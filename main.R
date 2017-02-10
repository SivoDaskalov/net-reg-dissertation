setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("batch_tools.R")

n = 100
transFactorsCount = 200
regulatedGenesPerTF = 10

batchResults = runBatch(n, transFactorsCount, regulatedGenesPerTF)
unpackBatchResults(batchResults)

# cl = makeCluster(6, type = "PSOCK", port=10101)
# clusterSetRNGStream(cl, 0)
# startTime <- proc.time()
# grace = grace(Xtr, Ytr[,1], Xtu, Ytu[,1], L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10, parallel = TRUE, cl = cl)
# graceElapsed <- (proc.time() - startTime)[3]
# stopCluster(cl)
# save.image("LiLi2008Big.rda")
# load("LiLi2008Big.rda")