lasso = function(Y, X, Ytu, Xtu, lambdas, K){
  cvLasso = cv.glmnet(y = Ytu, x = as.matrix(Xtu), lambda = lambdas, nfolds = 10)
  cvLambda = cvLasso$lambda.min
  lassoFit = glmnet(y = Y, x = as.matrix(X), lambda = cvLambda)
  return(list(lambda.min = cvLambda, coefficients = lassoFit$beta, fit = lassoFit, tuning = cvLasso))
}
