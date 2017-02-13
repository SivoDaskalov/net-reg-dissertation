setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("batch_tools.R")

n = 100
transFactorsCount = 20
regulatedGenesPerTF = 10

# batchResults = runBatch(n, transFactorsCount, regulatedGenesPerTF, methods = "grace")
batchResults = runBatch(n, transFactorsCount, regulatedGenesPerTF)
unpackBatchResults(batchResults)
# save.image("data/N100P220.RData")
