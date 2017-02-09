setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("dataset_tools.R")
source("evaluation_tools.R")
source("lasso.R")

batchLasso = function(Ytu, Xtu, Ytr, Xtr, Yts, Xts, lambdas, Betas){
  cases = list()
  for(i in 1:nrow(Betas)){
    lassoRes = lasso(Ytr[,i], Xtr, Ytu[,i], Xtu, lambda = lambdas, K = 10)
    prediction <- predict(object=lassoRes$fit, as.matrix(Xts), type="response")
    errors = evalErrors(Yts[,i],prediction)
    stat = evalBetaStatistics(trueBeta = as.numeric(Betas[i,]), betaHat = as.numeric(unlist(lassoRes$coefficients)))
    cases[[length(cases)+1]] = list(model = lassoRes, prediction = prediction, errors = errors$errors, mse = errors$mse, stderr = errors$stderr,
                                    cor = stat$cor, prec = stat$prec, spec = stat$spec, sens = stat$sens)
  }
  
  summary = c()
  for(i in 1:length(cases)){
    tmp = c(case = i, sens = cases[[i]]$sens, spec = cases[[i]]$spec, prec = cases[[i]]$prec,  
            cor = cases[[i]]$cor, mse = cases[[i]]$mse, stderr = cases[[i]]$stderr)
    summary <- rbind(summary, tmp)
  }
  rownames(summary) <- NULL
  
  return(list(cases = cases, summary = summary))
}

runBatch = function(n, factors, genesPerFactor){
  ds = generateDatasets(n, factors, genesPerFactor)
  
  models = list()
  
  lassoLambdaGrid = 10 ^ seq(from = -2, by = 1, length = 6)
  models$lasso = batchLasso(ds$Ytu, ds$Xtu, ds$Ytr, ds$Xtr, ds$Yts, ds$Xts, lambdas = lassoLambdaGrid, ds$Betas)
  
  result = list()
  result$L = ds$L
  result$betas = ds$Betas
  result$datasets = list(Ytu = ds$Ytu, Xtu = ds$Xtu, Ytr = ds$Ytr, Xtr = ds$Xtr, Yts = ds$Yts, Xts = ds$Xts)
  result$models = models
  return(result)
}
