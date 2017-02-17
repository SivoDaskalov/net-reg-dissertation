
lasso = function(X, Y, Xtu, Ytu, lambdas, K){
  alpha = 1
  cvLasso = cv.glmnet(y = Ytu, x = as.matrix(Xtu), lambda = lambdas, nfolds = K, alpha = alpha)
  cvLambda = cvLasso$lambda.min
  lassoFit = glmnet(y = Y, x = as.matrix(X), lambda = cvLambda, alpha = alpha)
  return(list(lambda.min = cvLambda, coefficients = lassoFit$beta, fit = lassoFit, tuning = cvLasso))
}

batchLasso = function(Xtu, Ytu, Xtr, Ytr, Xts, Yts, lambdas, Betas){
  models = list()
  for(i in 1:nrow(Betas)){
    models[[i]] = lasso(Xtr, Ytr[,i], Xtu, Ytu[,i], lambda = lambdas, K = 10)
  }
  return(batchEvaluateModels(Xts, Yts, models, Betas))
}