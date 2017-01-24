
generateBetaVector = function(TF1, TF2, TF3, TF4, denominator, negative, positive, unrelatedTransFactorCount){
  EL1 = c(TF1, rep(-TF1/denominator, negative), rep(TF1/denominator, positive))
  EL2 = c(TF2, rep(-TF2/denominator, negative), rep(TF2/denominator, positive))
  EL3 = c(TF3, rep(-TF3/denominator, negative), rep(TF3/denominator, positive))
  EL4 = c(TF4, rep(-TF4/denominator, negative), rep(TF4/denominator, positive))
  ELU = rep(0, unrelatedTransFactorCount*(1+positive+negative))
  return(c(EL1, EL2, EL3, EL4, ELU))
}

createBetaFrame = function(){
  M1 = generateBetaVector(TF1 = 5, TF2 = -5, TF3 = 3, TF4 = -3, denominator = sqrt(10), negative = 0, positive = 10, unrelatedTransFactorCount = 196)
  M2 = generateBetaVector(TF1 = 5, TF2 = -5, TF3 = 3, TF4 = -3, denominator = sqrt(10), negative = 3, positive = 7, unrelatedTransFactorCount = 196)
  M3 = generateBetaVector(TF1 = 5, TF2 = -5, TF3 = 3, TF4 = -3, denominator = 10, negative = 0, positive = 10, unrelatedTransFactorCount = 196)
  M4 = generateBetaVector(TF1 = 5, TF2 = -5, TF3 = 3, TF4 = -3, denominator = 10, negative = 3, positive = 7, unrelatedTransFactorCount = 196)
  betas = data.frame(rbind(M1,M2,M3,M4))
  return(betas)
}

generateExpressionLevels = function(transFactorsCount, regulatedGenesPerTF){
  TF = rnorm(n = transFactorsCount, mean = 0, sd = 1)
  expressionLevels = c()
  for(i in 1:length(TF)){
    expressionLevels = c(expressionLevels, TF[i], rnorm(regulatedGenesPerTF, 0.7*TF[i], sqrt(0.51)))
  }
  return(expressionLevels)
}

generateExpressionLevelsFrame = function(n, transFactorsCount, regulatedGenesPerTF){
  expressionLevelsFrame = c()
  for(i in 1:n){
    expressionLevelsFrame = rbind(expressionLevelsFrame, generateExpressionLevels(transFactorsCount, regulatedGenesPerTF))
  }
  return(data.frame(expressionLevelsFrame))
}

simulateResponseValues = function(X, betas){
  betas = as.numeric(betas)
  noiseVar = sum((betas-mean(betas))^2)/4
  noise = rnorm(nrow(X), mean = 0, sd = sqrt(noiseVar))
  
  weighedX = sweep(x = X, 2, betas, "*")
  sumWeighedX = rowSums(weighedX)
  
  Y = sumWeighedX + noise
  return(Y)
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
  degrees<-rep(c(regulatedGenesPerTF, rep(1, regulatedGenesPerTF)), transFactorsCount)^((gamma+1)/2)
  
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

normalize = function(X, Y){
  Xmu<-apply(X, 2, mean)
  Xsd<-sqrt(apply(X, 2, var))
  for(i in 1:length(X)){
    X[,i] = (X[,i] - Xmu[i])/Xsd[i]
  }
  
  for(i in 1:length(Y)){
    Y[,i] = Y[,i] - mean(Y[,i])
  }
  
  return(list(x = X, y = Y))
}

generateAndNormalize = function(n, transFactorsCount, regulatedGenesPerTF){
  set.seed(0)
  betas = createBetaFrame()
  L = generateNetwork(transFactorsCount, regulatedGenesPerTF)
  X = generateExpressionLevelsFrame(n, transFactorsCount, regulatedGenesPerTF)
  Y = data.frame(cbind(simulateResponseValues(X, betas[1,]), simulateResponseValues(X, betas[2,]), 
                       simulateResponseValues(X, betas[3,]), simulateResponseValues(X, betas[4,])))
  normalized = normalize(X, Y)
  return(list(x = normalized[[1]], y = normalized[[2]], l = L))
}

generated = generateAndNormalize(n = 100, transFactorsCount = 200, regulatedGenesPerTF = 10)
X = generated[[1]]
Y = generated[[2]]
L = generated[[3]]
