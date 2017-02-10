lasso = function(Y, X, Ytu, Xtu, lambdas, K){
  alpha = 1
  cvLasso = cv.glmnet(y = Ytu, x = as.matrix(Xtu), lambda = lambdas, nfolds = K, alpha = alpha)
  cvLambda = cvLasso$lambda.min
  lassoFit = glmnet(y = Y, x = as.matrix(X), lambda = cvLambda, alpha = alpha)
  return(list(lambda.min = cvLambda, coefficients = lassoFit$beta, fit = lassoFit, tuning = cvLasso))
}
