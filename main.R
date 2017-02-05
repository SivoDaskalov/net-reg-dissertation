setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("dataset_tools.R")
source("evaluation_tools.R")
source("grace.R")

# originalResultsTF20 = list(sens = 0.75, spec = 0.983, prec = 0.917, cor = 0.838, err = 485.283)
# optimizedResultsTF200 = list(sens = 1, spec = 0.231, prec = 0.025, cor = 0.623, err = 338.914)

n = 100
transFactorsCount = 200
regulatedGenesPerTF = 10

set.seed(0)
L = generateNetwork(transFactorsCount, regulatedGenesPerTF)
Betas = generateBetas(transFactorsCount, regulatedGenesPerTF)

Xtr = generateDataset(n, transFactorsCount, regulatedGenesPerTF)
Ytr = simulateResponse(Xtr, Betas)

Xtu = generateDataset(n, transFactorsCount, regulatedGenesPerTF)
Ytu = simulateResponse(Xtu, Betas)

Xts = generateDataset(n, transFactorsCount, regulatedGenesPerTF)
Yts = simulateResponse(Xts, Betas)

lambdaGrid = 10 ^ seq(from = -2, by = 1, length = 6)

# startTime <- proc.time()
# graceSeq = grace(Xtr, Ytr[,1], Xtu, Ytu[,1], L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10)
# seqTime <- proc.time() - startTime

cl = makeCluster(6, type = "PSOCK", port=10101)
startTime <- proc.time()
gracePar = grace(Xtr, Ytr[,1], Xtu, Ytu[,1], L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10, parallel = TRUE, cl = cl)
parTime <- proc.time() - startTime
parTime <- parTime[3]

# stopCluster(cl)

# prediction <- predict(object=graceRes$fit, as.matrix(Xts), type="response")
# errors = prediction - Yts[,1]
# std = function(x) sd(x)/sqrt(length(x))
# mse = mean(errors^2)
# stderr = std(errors)
# 
# betaHat = as.numeric(unlist(graceRes$coefficients$beta))
# trueBeta = as.numeric(Betas[1,])
# stat = evalStatistics(trueBeta = as.numeric(Betas[1,]), betaHat = as.numeric(unlist(graceRes$coefficients$beta)))
# tuning = graceRes$parameters$errors
# 
# betaHat

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