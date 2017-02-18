
grace <- function(xtr, ytr, xtu, ytu, network, degrees, lambda.1 = 0, lambda.2 = 0, adjustments = NULL, weights = NULL, k = 10, norun = FALSE){
  
  # --- General Adjusted Grace Model is as follows ---
  # variable b(p)
  # pen = b(netwk(:,1))./deg(netwk(:,1))-a(:).*b(netwk(:,2))./deg(netwk(:,2))
  # minimize(sum_square(y-x*b)+lam1*norm(b,1)+lam2*sum((pen.^2).*wt(:)));
  
  title = "Adaptive Grace"
  tuning.params = list(lam1 = lambda.1, lam2 = lambda.2)
  const.vars = list(netwk = network, deg = sqrt(degrees), wt = weights, a = adjustments)
  
  networkPenalty = "pen = b(netwk(:,1))./deg(netwk(:,1))-a(:).*b(netwk(:,2))./deg(netwk(:,2))"
  totalPenalty = "minimize(sum_square(y-x*b)+lam1*norm(b,1)+lam2*sum((pen.^2).*wt(:)));"
  
  if(is.null(weights) || max(weights) == min(weights) && weights[[1]]==1){
    # Edge weights assumed to be 1 and can be omitted from the calculation
    totalPenalty = "minimize(sum_square(y-x*b)+lam1*norm(b,1)+lam2*sum_square(pen));"
    const.vars[["wt"]] <- NULL
  }
  
  if(is.null(adjustments) || max(adjustments) == min(adjustments) && adjustments[[1]]==1){
    # Edge adjustments vector is missing - vanilla Grace is used
    networkPenalty = "pen = b(netwk(:,1))./deg(netwk(:,1))-b(netwk(:,2))./deg(netwk(:,2))"
    const.vars[["a"]] <- NULL
    title = "Grace"
  }
  
  graceConvexCode = paste("variable b(p)", networkPenalty, totalPenalty, sep="; ")
  
  return(cvxTuneAndTrain(xtr = xtr, ytr = ytr, xtu = xtu, ytu = ytu, cvxcode = graceConvexCode, tuning.params = tuning.params, const.vars = const.vars, title = title, norun = FALSE))
}

batchGrace = function(xtu, ytu, xtr, ytr, xts, yts, network, degrees, lambda.1, lambda.2, betas, k=10){
  models = list()
  for(i in 1:nrow(betas)){
    cat(timestamp("Setup "), i, "\n")
    models[[i]] = grace(xtr = xtr, ytr = ytr[,i], xtu = xtu, ytu = ytu[,i], 
                        network = network, degrees = degrees, k = k,
                        lambda.1 = lambda.1, lambda.2 = lambda.2)
  }
  return(batchEvaluateModels(xts, yts, models, betas))
}

batchAGrace = function(xtu, ytu, xtr, ytr, xts, yts, network, degrees, lambda.1, lambda.2, enetModels, betas, k=10){
  models = list()
  for(i in 1:nrow(betas)){
    cat(timestamp("Setup "), i, "\n")
    # Estimate initial betas
    if(ncol(xtu) < nrow(xtu)){
      # Take initial estimates based on OLSE
      lmData = data.frame(Y = ytu[,i], xtu)
      lmFit = lm(Y ~ .-1, data = lmData)
      b0 = lmFit$coefficients
    } else{
      # Take initial estimates based on ENet
      b0 = enetModels$cases[[i]]$coefficients
    }
    
    # Setup adjustments vector
    adj = apply(network, 1, signAdjustment, b = b0)
    
    models[[i]] = grace(xtr = xtr, ytr = ytr[,i], xtu = xtu, ytu = ytu[,i], 
                        network = network, degrees = degrees, k = k,
                        lambda.1 = lambda.1, lambda.2 = lambda.2,
                        adjustments = adj)
  }
  return(batchEvaluateModels(xts, yts, models, betas))
}

signAdjustment = function(edge, b){
  if(b[edge[1]] > 0 && b[edge[2]] > 0 ){
    return(1)
  } else{
    return(-1)
  }
}
