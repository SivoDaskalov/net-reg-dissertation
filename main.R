setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("dataset_tools.R")
source("evaluation_tools.R")
source("grace.R")

n = 100
transFactorsCount = 20
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
grace = grace(Ytr[,1], Xtr, Ytu[,1], Xtu, L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10)

prediction <- predict(object=grace$fit, as.matrix(Xts), type="response")
mse = mean((prediction - Yts[,1])^2)

betaHat = as.numeric(unlist(grace$coefficients$beta))
trueBeta = as.numeric(Betas[1,])
evalStatistics(trueBeta = as.numeric(Betas[1,]), betaHat = as.numeric(unlist(grace$coefficients$beta)))
