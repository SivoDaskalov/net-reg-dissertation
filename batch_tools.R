setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("dataset_tools.R")
source("evaluation_tools.R")
source("methods/lasso.R")
source("methods/enet.R")
source("methods/grace.R")

runBatch = function(n, factors, genesPerFactor, methods){
  set.seed(0)
  
  if(missing(methods)){
    methods = c("lasso", "enet")
  }
  
  start = proc.time()
  cat(timestamp("Generating datasets"), "\n")
  ds = generateDatasets(n, factors, genesPerFactor)
  
  models = list()
  
  if("lasso" %in% methods){
    cat(timestamp("Fitting Lasso models"), "\n")
    lassoLambdaGrid = 10 ^ seq(from = -2, by = 1, length = 6)
    models$lasso = batchLasso(ds$Ytu, ds$Xtu, ds$Ytr, ds$Xtr, ds$Yts, ds$Xts, lambdas = lassoLambdaGrid, ds$Betas)
  }
  
  if("enet" %in% methods){
    cat(timestamp("Fitting Elastic Net models"), "\n")
    enetLambdaGrid = 10 ^ seq(from = -2, by = 1, length = 6)
    models$enet = batchEnet(ds$Ytu, ds$Xtu, ds$Ytr, ds$Xtr, ds$Yts, ds$Xts, lambdas = enetLambdaGrid, ds$Betas)
  }
  
  result = list()
  result$L = ds$L
  result$betas = ds$Betas
  result$datasets = list(Ytu = ds$Ytu, Xtu = ds$Xtu, Ytr = ds$Ytr, Xtr = ds$Xtr, Yts = ds$Yts, Xts = ds$Xts)
  result$models = models
  result$timeElapsed = (proc.time() - start)[3]
  return(result)
}

batchLasso = function(Ytu, Xtu, Ytr, Xtr, Yts, Xts, lambdas, Betas){
  models = list()
  for(i in 1:nrow(Betas)){
    models[[i]] = lasso(Ytr[,i], Xtr, Ytu[,i], Xtu, lambda = lambdas, K = 10)
  }
  return(batchEvaluateModels(Yts, Xts, models, Betas))
}

batchEnet = function(Ytu, Xtu, Ytr, Xtr, Yts, Xts, lambdas, Betas){
  models = list()
  for(i in 1:nrow(Betas)){
    models[[i]] = enet(Ytr[,i], Xtr, Ytu[,i], Xtu, lambda = lambdas, K = 10)
  }
  return(batchEvaluateModels(Yts, Xts, models, Betas))
}

batchEvaluateModels = function(Yts, Xts, models, Betas){
  summary = c()
  predictions = c()
  for(i in 1:nrow(Betas)){
    model = models[[i]]
    
    prediction = predict(object=model$fit, as.matrix(Xts), type="response")
    predictions = cbind(predictions, prediction)
    
    errors = evalErrors(Yts[,i],prediction)
    stat = evalBetaStatistics(trueBeta = as.numeric(Betas[i,]), betaHat = as.numeric(unlist(model$coefficients)))
    summary = rbind(summary, c(case = i, sens = stat$sens, spec = stat$spec, prec = stat$prec, 
                               cor = stat$cor, mse = errors$mse, stderr = errors$stderr))
  }
  rownames(summary) = NULL
  colnames(predictions) = seq(1,length(models))
  return(list(cases = models, summary = summary, predictions = predictions))
}

unpackBatchResults = function (batchResults){
  L <<- batchResults$L
  betas <<- batchResults$betas
  Xtu <<- batchResults$datasets$Xtu
  Ytu <<- batchResults$datasets$Ytu
  Xtr <<- batchResults$datasets$Xtr
  Ytr <<- batchResults$datasets$Ytr
  Xts <<- batchResults$datasets$Xts
  Yts <<- batchResults$datasets$Yts
  
  models <<- batchResults$models
  methodNames = attributes(models)$names
  for(i in 1:length(methodNames)){
    methodName = methodNames[i]
    currentMethod = models[[methodName]]
    assign(paste(methodName, "Models", sep = ""), currentMethod$cases, envir = .GlobalEnv)
    assign(paste(methodName, "Summary", sep = ""), currentMethod$summary, envir = .GlobalEnv)
    assign(paste(methodName, "Predictions", sep = ""), currentMethod$predictions, envir = .GlobalEnv)
  }
  timeElapsed <<- batchResults$timeElapsed
}

timestamp = function(x){
  return (paste(format(Sys.time(), "%X"), "-", x)[1])
}