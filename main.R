setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("dataset_tools.R")
source("grace.R")

n = 100
transFactorsCount = 200
regulatedGenesPerTF = 10

set.seed(0)
training = generateAndNormalize(n, transFactorsCount, regulatedGenesPerTF)
Xtr = training[[1]]
Ytr = training[[2]]
L = training[[3]]

tuning = generateAndNormalize(n, transFactorsCount, regulatedGenesPerTF)
Xtu = tuning[[1]]
Ytu = tuning[[2]]

test = generateAndNormalize(n, transFactorsCount, regulatedGenesPerTF)
Xts = test[[1]]
Yts = test[[2]]
Betasts = test[[4]]

lambdaGrid = 10 ^ seq(from = -2, by = 1, length = 6)
grace = grace(Ytr[,1], Xtr, Ytu[,1], Xtu, L, lambdaGrid, lambdaGrid, lambdaGrid, K = 10)

prediction <- predict(object=grace$fit, as.matrix(Xts), type="response")
mse = mean((prediction - Yts[,1])^2)
coefs = grace$coefficients$beta
trueBetas = 4 * (1 + regulatedGenesPerTF)
sens = sum(coefs[1:trueBetas] != 0) / trueBetas
spec = sum(coefs[trueBetas+1:(length(coefs)-trueBetas)] == 0) / (length(coefs) - trueBetas)
grace$parameters$parameterMin
