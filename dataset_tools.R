dataTools <- new.env()

dataTools$generateBetaVector = function(TF1, TF2, TF3, TF4, denominator, negative, positive, unrelatedTransFactorCount){
  EL1 = c(TF1, rep(-TF1/denominator, negative), rep(TF1/denominator, positive))
  EL2 = c(TF2, rep(-TF2/denominator, negative), rep(TF2/denominator, positive))
  EL3 = c(TF3, rep(-TF3/denominator, negative), rep(TF3/denominator, positive))
  EL4 = c(TF4, rep(-TF4/denominator, negative), rep(TF4/denominator, positive))
  ELU = rep(0, unrelatedTransFactorCount*(1+positive+negative))
  return(c(EL1, EL2, EL3, EL4, ELU))
}

dataTools$generateObservation = function(transFactorsCount, regulatedGenesPerTF){
  TF = rnorm(n = transFactorsCount, mean = 0, sd = 1)
  observation = c()
  for(i in 1:length(TF)){
    observation = c(observation, TF[i], rnorm(regulatedGenesPerTF, 0.7*TF[i], sqrt(0.51)))
  }
  return(observation)
}

dataTools$generateExpressionLevelsFrame = function(n, transFactorsCount, regulatedGenesPerTF){
  expressionLevelsFrame = c()
  for(i in 1:n){
    expressionLevelsFrame = rbind(expressionLevelsFrame, dataTools$generateObservation(transFactorsCount, regulatedGenesPerTF))
  }
  return(data.frame(expressionLevelsFrame))
}

dataTools$simulateResponseVector = function(X, betas){
  betas = as.numeric(betas)
  noiseVar = sum((betas-mean(betas))^2)/4
  noise = rnorm(nrow(X), mean = 0, sd = sqrt(noiseVar))
  
  weighedX = sweep(x = X, 2, betas, "*")
  sumWeighedX = rowSums(weighedX)
  
  Y = sumWeighedX + noise
  return(Y)
}

generateDataset = function(n, transFactorsCount, regulatedGenesPerTF){
  X = dataTools$generateExpressionLevelsFrame(n, transFactorsCount, regulatedGenesPerTF)
  
  # Normalize
  Xmu<-apply(X, 2, mean)
  Xsd<-sqrt(apply(X, 2, var))
  for(i in 1:length(X)){
    X[,i] = (X[,i] - Xmu[i])/Xsd[i]
  }
  
  return(X)
}

simulateResponse = function(X, betas){
  Y = c()
  for(i in 1:nrow(betas)){
    Y = cbind(Y, dataTools$simulateResponseVector(X, betas[i,]))
  }
  
  # Normalize
  Y = data.frame(Y)
  for(i in 1:length(Y)){
    Y[,i] = Y[,i] - mean(Y[,i])
  }
  
  return(Y)
}

generateBetas = function(transFactorsCount, regulatedGenesPerTF){
  M1 = dataTools$generateBetaVector(TF1 = 5, TF2 = -5, TF3 = 3, TF4 = -3, denominator = sqrt(10), 
                                    negative = 0, positive = regulatedGenesPerTF, unrelatedTransFactorCount = transFactorsCount - 4)
  M2 = dataTools$generateBetaVector(TF1 = 5, TF2 = -5, TF3 = 3, TF4 = -3, denominator = sqrt(10), 
                                    negative = 3, positive = regulatedGenesPerTF - 3, unrelatedTransFactorCount = transFactorsCount - 4)
  M3 = dataTools$generateBetaVector(TF1 = 5, TF2 = -5, TF3 = 3, TF4 = -3, denominator = 10, 
                                    negative = 0, positive = regulatedGenesPerTF, unrelatedTransFactorCount = transFactorsCount - 4)
  M4 = dataTools$generateBetaVector(TF1 = 5, TF2 = -5, TF3 = 3, TF4 = -3, denominator = 10, 
                                    negative = 3, positive = regulatedGenesPerTF - 3, unrelatedTransFactorCount = transFactorsCount - 4)
  betas = data.frame(rbind(M1,M2,M3,M4))
  return(betas)
}

generateNetwork = function(transFactorsCount, regulatedGenesPerTF){
  network = matrix(0, transFactorsCount * regulatedGenesPerTF, 2)
  for(i in 0:transFactorsCount-1){
    for(j in 1:regulatedGenesPerTF){
      network[i * regulatedGenesPerTF + j, 1] = i * (regulatedGenesPerTF + 1) + 1
      network[i * regulatedGenesPerTF + j, 2] = i * (regulatedGenesPerTF + 1) + j + 1
    }
  }
  
  gamma = 0
  degrees = rep(c(regulatedGenesPerTF, rep(1, regulatedGenesPerTF)), transFactorsCount)^((gamma+1)/2)
  
  p = transFactorsCount * ( 1 + regulatedGenesPerTF)
  L = matrix(0, p, p)
  for(i in 1:nrow(network)){
    # w(u,v) assumed to be 1
    u = network[i,1]
    v = network[i,2]
    if(u == v){
      L[u,v] = 1 - 1/degrees[u]
    } else{
      L[u,v] = - 1/sqrt(degrees[u]*degrees[v])
      L[v,u] = - 1/sqrt(degrees[u]*degrees[v])
    }
  }
  
  return(L)
}

generateDatasets = function(n, factors, genesPerFactor){
  L = generateNetwork(factors, genesPerFactor)
  Betas = generateBetas(factors, genesPerFactor)
  
  Xtu = generateDataset(n, factors, genesPerFactor)
  Ytu = simulateResponse(Xtu, Betas)
  
  Xtr = generateDataset(n, factors, genesPerFactor)
  Ytr = simulateResponse(Xtr, Betas)
  
  Xts = generateDataset(n, factors, genesPerFactor)
  Yts = simulateResponse(Xts, Betas)
  
  return(list(Xtu = Xtu, Ytu = Ytu, Xtr = Xtr, Ytr = Ytr, Xts = Xts, Yts = Yts, L = L, Betas = Betas))
}