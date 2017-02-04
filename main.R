setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("dataset_tools.R")
source("evaluation_tools.R")
source("grace.R")

originalResultsTF20 = list(sens = 0.75, spec = 0.983, prec = 0.917, cor = 0.838, err = 485.283)
optimizedResultsTF200 = list(sens = 1, spec = 0.231, prec = 0.025, cor = 0.623, err = 338.914)
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
graceRes = grace(Xtr, Ytr[,1], Xtu, Ytu[,1], L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10)

prediction <- predict(object=graceRes$fit, as.matrix(Xts), type="response")
error = mean((prediction - Yts[,1])^2)

betaHat = as.numeric(unlist(graceRes$coefficients$beta))
trueBeta = as.numeric(Betas[1,])
stat = evalStatistics(trueBeta = as.numeric(Betas[1,]), betaHat = as.numeric(unlist(graceRes$coefficients$beta)))
tuning = graceRes$parameters$errors



save.image("LiLi2008Big.rda")
load("LiLi2008Big.rda")
