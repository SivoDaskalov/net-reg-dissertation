setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("grace.R")
source("batch_tools.R")

n = 100
transFactorsCount = 200
regulatedGenesPerTF = 10

batch = runBatch(n, transFactorsCount, regulatedGenesPerTF)

L = batch$L
betas = batch$betas
Xtu = batch$datasets$Xtu
Ytu = batch$datasets$Ytu
Xtr = batch$datasets$Xtr
Ytr = batch$datasets$Ytr
Xts = batch$datasets$Xts
Yts = batch$datasets$Yts

models = batch$models

lassoModels = models$lasso$cases
lassoSummary = models$lasso$summary

# set.seed(0)
# L = generateNetwork(transFactorsCount, regulatedGenesPerTF)
# Betas = generateBetas(transFactorsCount, regulatedGenesPerTF)
# 
# Xtu = generateDataset(n, transFactorsCount, regulatedGenesPerTF)
# Ytu = simulateResponse(Xtu, Betas)
# 
# Xtr = generateDataset(n, transFactorsCount, regulatedGenesPerTF)
# Ytr = simulateResponse(Xtr, Betas)
# 
# Xts = generateDataset(n, transFactorsCount, regulatedGenesPerTF)
# Yts = simulateResponse(Xts, Betas)
# 
# lambdaGrid = 10 ^ seq(from = -2, by = 1, length = 6)

# startTime <- proc.time()
# graceSeq = grace(Xtr, Ytr[,1], Xtu, Ytu[,1], L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10)
# seqTime <- proc.time() - startTime
# seqTime <- seqTime[3]

# cl = makeCluster(6, type = "PSOCK", port=10101)
# clusterSetRNGStream(cl, 0)
# startTime <- proc.time()
# gracePar = grace(Xtr, Ytr[,1], Xtu, Ytu[,1], L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10, parallel = TRUE, cl = cl)
# parTime <- proc.time() - startTime
# parTime <- parTime[3]

# stopCluster(cl)

# graceRes = gracePar
# prediction <- predict(object=graceRes$fit, as.matrix(Xts), type="response")
# errors = evalErrors(Yts[,1],prediction)
# betaHat = as.numeric(unlist(graceRes$coefficients$beta))
# trueBeta = as.numeric(Betas[1,])
# stat = evalBetaStatistics(trueBeta = as.numeric(Betas[1,]), betaHat = as.numeric(unlist(graceRes$coefficients$beta)))
# tuning = graceRes$parameters$errors

# save.image("LiLi2008Big.rda")
# load("LiLi2008Big.rda")

# startTime <- proc.time()
# graceSeq = cvGrace(Xtu, Ytu[,1], L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10)
# seqTime <- proc.time() - startTime
# gracePar = pcvGrace(Xtu, Ytu[,1], L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10)
# parTime <- proc.time() - seqTime
# 
# seqErrors = graceSeq$errors
# parErrors = gracePar$errors