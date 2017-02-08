setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("dataset_tools.R")
source("evaluation_tools.R")
source("lasso.R")

batchLasso = function(Ytu, Xtu, Ytr, Xtr, Yts, Xts, lambdas, Betas){
  
  for(i in 1:nrow(Betas)){
    Ytu[,i]
    Betas[i,]
    lassoRes = lasso(Ytr[,i], Xtr, Ytu[,i], Xtu, lambda = lambdas, K = 10)
    prediction <- predict(object=lassoRes$fit, as.matrix(Xts), type="response")
    
  }
  return(list())
}

runBatch = function(n, factors, genesPerFactor){
  ds = generateDatasets(n, factors, genesPerFactor)
  
  lassoLambdaGrid = 10 ^ seq(from = -2, by = 1, length = 6)
  lassoRes = batchLasso(ds$Ytu, ds$Xtu, ds$Ytr, ds$Xtr, ds$Yts, ds$Xts, lambdas = lassoLambdaGrid, ds$Betas)
  
  models = list()
  models$lasso = lassoRes
  
  result = list()
  result$L = ds$L
  result$betas = ds$Betas
  result$datasets = list(Ytu = ds$Ytu, Xtu = ds$Xtu, Ytr = ds$Ytr, Xtr = ds$Xtr, Yts = ds$Yts, Xts = ds$Xts)
  result$models = models
  return(result)
}
