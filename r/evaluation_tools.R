
evalErrors = function(actual, predicted){
  errors = predicted - actual
  mse = mean(errors^2)
  stderr = sd(errors)/sqrt(length(errors))
  return(list(errors = errors, mse = mse, stderr = stderr))
}

evalBetaStatistics = function(trueBeta, betaHat, trueBetaIndices){
  if(missing(trueBetaIndices)) {
    # Assume beta model as defined in Li and Li 2008
    trueTransFactors = 4
    genesPerFactor = 10
    trueBetaIndices = 1:(trueTransFactors*(1+genesPerFactor))
  } 
  
  correlation = cor(x = betaHat, y = trueBeta)
  sensitivity = sum(betaHat[trueBetaIndices] != 0) / length(trueBetaIndices)
  specificity = sum(betaHat[(length(trueBetaIndices) + 1) : (length(betaHat))] == 0) / (length(betaHat) - length(trueBetaIndices))
  precision = sum(betaHat[trueBetaIndices] != 0) / sum(betaHat != 0)
  return(list(sens = sensitivity, spec = specificity, prec = precision, cor = correlation))
}
