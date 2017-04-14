
enet = function(X, Y, Xtu, Ytu, lambdas, K){
  alpha = 0.5
  cvEnet = cv.glmnet(y = Ytu, x = as.matrix(Xtu), lambda = lambdas, nfolds = K, alpha = alpha)
  cvLambda = cvEnet$lambda.min
  enetFit = glmnet(y = Y, x = as.matrix(X), lambda = cvLambda, alpha = alpha)
  return(list(lambda.min = cvLambda, coefficients = enetFit$beta, fit = enetFit, tuning = cvEnet))
}

batchEnet = function(Xtu, Ytu, Xtr, Ytr, Xts, Yts, lambdas, Betas){
  models = list()
  for(i in 1:nrow(Betas)){
    models[[i]] = enet(Xtr, Ytr[,i], Xtu, Ytu[,i], lambda = lambdas, K = 10)
  }
  return(batchEvaluateModels(Xts, Yts, models, Betas))
}
