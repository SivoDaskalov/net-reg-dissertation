enet = function(X, Y, Xtu, Ytu, lambdas, K){
  alpha = 0.5
  cvEnet = cv.glmnet(y = Ytu, x = as.matrix(Xtu), lambda = lambdas, nfolds = 10, alpha = alpha)
  cvLambda = cvEnet$lambda.min
  enetFit = glmnet(y = Y, x = as.matrix(X), lambda = cvLambda, alpha = alpha)
  return(list(lambda.min = cvLambda, coefficients = enetFit$beta, fit = enetFit, tuning = cvEnet))
}
