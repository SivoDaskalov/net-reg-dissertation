setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("dataset_tools.R")
source("evaluation_tools.R")
source("methods/lasso.R")
source("methods/enet.R")
source("methods/grace.R")
source("methods/gblasso.R")

runBatch = function(n, factors, genesPerFactor, methods){
  set.seed(0)
  
  if(missing(methods)){
    methods = c("lasso", "enet", "grace", "agrace", "gblasso")
  }
  
  if("agrace" %in% methods && !("enet" %in% methods)){
    methods <- c(methods, "enet")
  }
  
  start = proc.time()
  cat(timestamp("Generating datasets"), "\n")
  ds = generateDatasets(n, factors, genesPerFactor)
  
  models = list()
  
  if("lasso" %in% methods){
    cat(timestamp("Fitting Lasso models"), "\n")
    lassoLambdaGrid = 10 ^ seq(from = -2, by = 1, length = 6)
    models$lasso = batchLasso(ds$Xtu, ds$Ytu, ds$Xtr, ds$Ytr, ds$Xts, ds$Yts, lambdas = lassoLambdaGrid, ds$Betas)
  }
  
  if("enet" %in% methods){
    cat(timestamp("Fitting Elastic Net models"), "\n")
    enetLambdaGrid = 10 ^ seq(from = -2, by = 1, length = 6)
    models$enet = batchEnet(ds$Xtu, ds$Ytu, ds$Xtr, ds$Ytr, ds$Xts, ds$Yts, lambdas = enetLambdaGrid, ds$Betas)
  }
  
  if("grace" %in% methods){
    cat(timestamp("Fitting Grace models"), "\n")
    # grace.lambda.L = grace.lambda.1 = grace.lambda.2 = 10 ^ seq(from = -2, by = 1, length = 6)
    grace.lambda.L = grace.lambda.1 = grace.lambda.2 = 10 * seq(from = 1, by = 1, length = 10)
    models$grace = batchGrace(Xtu = ds$Xtu, Ytu = ds$Ytu, Xtr = ds$Xtr, Ytr = ds$Ytr, Xts = ds$Xts, Yts = ds$Yts, L = ds$L, 
                              lambda.L = grace.lambda.L, lambda.1 = grace.lambda.1, lambda.2 = grace.lambda.2, Betas = ds$Betas)
  }
  
  if("agrace" %in% methods){
    cat(timestamp("Fitting Adaptive Grace models"), "\n")
    grace.lambda.L = grace.lambda.1 = grace.lambda.2 = 10 * seq(from = 1, by = 1, length = 10)
    models$agrace = batchAGrace(Xtu = ds$Xtu, Ytu = ds$Ytu, Xtr = ds$Xtr, Ytr = ds$Ytr, Xts = ds$Xts, Yts = ds$Yts, 
                                L = ds$L, enetModels = models$enet, Betas = ds$Betas,
                                lambda.L = grace.lambda.L, lambda.1 = grace.lambda.1, lambda.2 = grace.lambda.2)
  }
  
  if("gblasso" %in% methods){
    cat(timestamp("Fitting GBLasso models"), "\n")
    models$gblasso = batchGBLasso(ds$Xtu, ds$Ytu, ds$Xtr, ds$Ytr, ds$Xts, ds$Yts, 
                                ds$edges, ds$degrees, ds$Betas)
  }
  
  result = list()
  result$L = ds$L
  result$edges = ds$edges
  result$degrees = ds$degrees
  result$betas = ds$Betas
  result$datasets = list(Ytu = ds$Ytu, Xtu = ds$Xtu, Ytr = ds$Ytr, Xtr = ds$Xtr, Yts = ds$Yts, Xts = ds$Xts)
  result$models = models
  result$timeElapsed = (proc.time() - start)[3]
  cat(timestamp("Batch fitting completed"), "\n")
  return(result)
}

batchLasso = function(Xtu, Ytu, Xtr, Ytr, Xts, Yts, lambdas, Betas){
  models = list()
  for(i in 1:nrow(Betas)){
    models[[i]] = lasso(Xtr, Ytr[,i], Xtu, Ytu[,i], lambda = lambdas, K = 10)
  }
  return(batchEvaluateModels(Xts, Yts, models, Betas))
}

batchEnet = function(Xtu, Ytu, Xtr, Ytr, Xts, Yts, lambdas, Betas){
  models = list()
  for(i in 1:nrow(Betas)){
    models[[i]] = enet(Xtr, Ytr[,i], Xtu, Ytu[,i], lambda = lambdas, K = 10)
  }
  return(batchEvaluateModels(Xts, Yts, models, Betas))
}

batchGrace = function(Xtu, Ytu, Xtr, Ytr, Xts, Yts, L, lambda.L, lambda.1, lambda.2, Betas){
  cores = min(6, detectCores())
  cluster = makeCluster(cores, type = "PSOCK", port=10101)
  clusterSetRNGStream(cluster, 0)

  models = list()
  for(i in 1:nrow(Betas)){
    models[[i]] = grace(X = Xtr, Y = Ytr[,i], Xtu = Xtu, Ytu = Ytu[,i], 
                        L = L, K = 10, parallel = TRUE, cl = cluster, 
                        lambda.L = lambda.L, lambda.1 = lambda.1, lambda.2 = lambda.2)
  }
  
  stopCluster(cluster)
  registerDoSEQ()
  return(batchEvaluateModels(Xts, Yts, models, Betas))
}

batchAGrace = function(Xtu, Ytu, Xtr, Ytr, Xts, Yts, L, lambda.L, lambda.1, lambda.2, Betas, enetModels){
  cores = min(6, detectCores())
  cluster = makeCluster(cores, type = "PSOCK", port=10101)
  clusterSetRNGStream(cluster, 0)

  models = list()
  for(i in 1:nrow(Betas)){
    models[[i]] = grace(X = Xtr, Y = Ytr[,i], Xtu = Xtu, Ytu = Ytu[,i], 
                        L = L, K = 10, parallel = TRUE, cl = cluster, enetFit = enetModels$cases[[i]],
                        lambda.L = lambda.L, lambda.1 = lambda.1, lambda.2 = lambda.2)
  }
  
  stopCluster(cluster)
  registerDoSEQ()
  return(batchEvaluateModels(Xts, Yts, models, Betas))
}

batchGBLasso = function(Xtu, Ytu, Xtr, Ytr, Xts, Yts, edges, degrees, Betas){
  wt = degrees^(1/2)
  models = list()
  for(i in 1:nrow(Betas)){
    # Currently training with the tuning datasets
    tuning <<- GBLasso(Xtu, Ytu[,i], netwk = edges, wt = wt)
    minMse = mse(Y = Yts[,i], X = Xts, b = tuning$betas[1,])
    solIdx = 1
    for(j in 2:tuning$nsol){
      curMse = mse(Y = Yts[,i], X = Xts, b = tuning$betas[j,])
      if(curMse < minMse){
        minMse = curMse
        solIdx = j
      }
    }
    tuning$coefficients = tuning$betas[solIdx,]
    model = list(fit = tuning, coefficients = tuning$betas[solIdx,])
    models[[i]] = model
  }
  return(batchEvaluateModels(Xts, Yts, models, Betas))
}

batchEvaluateModels = function(Xts, Yts, models, Betas){
  summary = c()
  predictions = c()
  for(i in 1:nrow(Betas)){
    model = models[[i]]
    
    prediction = predict(object=model$fit, as.matrix(Xts), type="response")
    predictions = cbind(predictions, prediction)
    
    errors = evalErrors(Yts[,i],prediction)
    stat = evalBetaStatistics(trueBeta = as.numeric(Betas[i,]), betaHat = as.numeric(unlist(model$coefficients)))
    summary = rbind(summary, c(case = i, sens = stat$sens, spec = stat$spec, prec = stat$prec, 
                               cor = stat$cor, mse = errors$mse, stderr = errors$stderr, 
                               nzero = sum(model$coefficients != 0)))
  }
  rownames(summary) = NULL
  colnames(predictions) = seq(1,length(models))
  return(list(cases = models, summary = summary, predictions = predictions))
}

unpackBatchResults = function (batchResults){
  L <<- batchResults$L
  degrees <<- batchResults$degrees
  edges <<- batchResults$edges
  betas <<- batchResults$betas
  Xtu <<- batchResults$datasets$Xtu
  Ytu <<- batchResults$datasets$Ytu
  Xtr <<- batchResults$datasets$Xtr
  Ytr <<- batchResults$datasets$Ytr
  Xts <<- batchResults$datasets$Xts
  Yts <<- batchResults$datasets$Yts
  
  
  models <<- batchResults$models
  methodNames = attributes(models)$names
  templateSetupSummary = matrix(NA, nrow = length(methodNames), ncol = ncol(models[[methodNames[1]]]$summary))
  colnames(templateSetupSummary) = colnames(models[[methodNames[1]]]$summary)
  
  totalSummary <<- as.data.frame(matrix(NA, nrow = length(methodNames) * nrow(betas), ncol = ncol(models[[methodNames[1]]]$summary)+1))
  colnames(totalSummary) <<- c("method", colnames(models[[methodNames[1]]]$summary))
  colnames(totalSummary)[2] <<- "setup"
  
  for(i in 1:nrow(betas)){
    current = paste("setup", i, "Summary", sep = "")
    assign(current, as.data.frame(templateSetupSummary), envir = .GlobalEnv)
  }
  for(i in 1:length(methodNames)){
    methodName = methodNames[i]
    currentMethod = models[[methodName]]
    assign(paste(methodName, "Models", sep = ""), currentMethod$cases, envir = .GlobalEnv)
    assign(paste(methodName, "Summary", sep = ""), currentMethod$summary, envir = .GlobalEnv)
    assign(paste(methodName, "Predictions", sep = ""), currentMethod$predictions, envir = .GlobalEnv)
    for(j in 1:nrow(currentMethod$summary)){
      tmp = get(paste("setup", j, "Summary", sep = ""), envir = .GlobalEnv)
      tmp[i,] = currentMethod$summary[j, ]
      tmp[i,1] = methodName
      assign(paste("setup", j, "Summary", sep = ""), tmp, envir = .GlobalEnv)
      totalSummary[(i-1)*nrow(currentMethod$summary)+j,] <<- c(method = methodName, round(currentMethod$summary[j, ], digits = 3))
    }
  }
  timeElapsed <<- batchResults$timeElapsed
}

timestamp = function(x){
  return (paste(format(Sys.time(), "%X"), "-", x)[1])
}