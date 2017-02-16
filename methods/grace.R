
cvGrace <- function(X, Y, L, lambda.L, lambda.1, lambda.2, K = 10, b0){
  X <- scale(X)
  Y <- Y - mean(Y)
  
  lambda.L <- unique(sort(lambda.L, decreasing = TRUE))
  lambda.1 <- unique(sort(lambda.1, decreasing = TRUE))
  lambda.2 <- unique(sort(lambda.2, decreasing = TRUE))
  
  p <- ncol(X)
  n <- nrow(X)
  errors = c()
  
  for(iL in 1:length(lambda.L)){
    lL <- lambda.L[iL]
    for(i2 in 1:length(lambda.2)){
      l2 <- lambda.2[i2]
      Lnew <- lL * L + l2 * diag(p)
      eL <- eigen(Lnew)
      if( sum(eL$values<=0)>0 ){next}
      S <- eL$vectors %*% sqrt(diag(eL$values))
      
      l2star <- 1
      l1star <- lambda.1
      Xstar <- rbind(X, sqrt(l2star) * t(S)) / sqrt(1 + l2star)
      Ystar <- c(Y, rep(0, p))
      gammastar <- l1star / sqrt(1 + l2star) / 2 / (n + p)
      
      cvres <- cv.glmnet(Xstar, Ystar, lambda = gammastar, intercept = FALSE, standardize = FALSE, nfolds = K)
      for(i1 in 1:length(gammastar)){
        if(!is.na(cvres$cvm[i1])){
          errors <- rbind( errors, c( lL, lambda.1[i1] , l2, cvres$cvm[i1], cvres$nzero[i1] ) )
        }
      }
    }
  }
  
  colnames(errors)=c("lambda.L","lambda.1","lambda.2","cvm","nzero")
  cvm = errors[,4]
  idxMin = which.min(cvm)
  sem = sd(cvm)/sqrt(length(cvm))
  idx1se = which.max(cvm < (cvm[idxMin] + sem))
  return(list(errors = errors, lambda.min = errors[idxMin,], lambda.1se = errors[idx1se,]))
}

grace <- function(X, Y, Xtu, Ytu, L, lambda.L, lambda.1 = 0, lambda.2 = 0, normalize.L = FALSE, K = 10, parallel = FALSE, cl, enetFit){
  if(!is.null(ncol(Y))){
    stop("Error: Y is not a vector.")
  }
  if(length(Y) != nrow(X)){
    stop("Error: Dimensions of X and Y do not match.")
  }
  if(!isSymmetric(L)){
    stop("Error: L is not a symmetric matrix.")
  }
  if(ncol(X) != ncol(L)){
    stop("Error: Dimensions of X and L do not match.")
  }
  if(min(lambda.L) < 0 | min(lambda.2) < 0 | min(lambda.1) < 0){
    stop("Error: Grace tuning parameters must be non-negative.")
  }
  if(min(lambda.L) == 0 & min(lambda.2) == 0){
    stop("Error: At least one of the grace tuning parameters must be positive.")
  }
  
  res = list()
  ori.Y <- Y
  ori.X <- X
  scale.fac <- attr(scale(X), "scaled:scale")
  X <- scale(X)     # Standardize X
  Y <- Y - mean(Y)  # Center Y
  n <- nrow(X)
  p <- ncol(X)
  
  if(missing(enetFit)){
    # Assuming regular grace
    b0 = rep(1, p)
  }else{
    # Assuming adaptive grace
    if(p<n){
      # Take initial estimates based on OLSE
      lmData = data.frame(Y = Y, Xtu)
      lmFit = lm(Y ~ .-1, data = Xtu)
      b0 = lmFit$coefficients
    } else{
      # Take initial estimates based on ENet
      b0 = enetFit$coefficients
    }
  }
  res$b0 = b0
  
  if(normalize.L){
    diag(L)[diag(L) == 0] <- 1
    L <- diag(1 / sqrt(diag(L))) %*% L %*% diag(1 / sqrt(diag(L)))  # Normalize L
  }
  
  # If more than one tuning parameter is provided, perform K-fold cross-validation  
  if((length(lambda.L) > 1) | (length(lambda.1) > 1) | (length(lambda.2) > 1)){
    if(missing(parallel) || parallel == FALSE){
      parameters <- cvGrace(Xtu, Ytu, L, lambda.L, lambda.1, lambda.2, K, b0)
    } else{
      ownCluster = FALSE
      if(missing(cl)){
        cat("Cluster not provided, initializing own")
        cores = min(6, detectCores())
        cl = makeCluster(cores, type = "PSOCK", port=10101, timeout=30)
        clusterSetRNGStream(cl, 0)
        ownCluster = TRUE
      }
      parameters <- pcvGrace(Xtu, Ytu, L, lambda.L, lambda.1, lambda.2, K, cl, b0)
      if(ownCluster == TRUE){
        stopCluster(cl)
        registerDoSEQ()
      }
    }
    res$tuning = parameters$errors
    res$lambda.min = parameters$lambda.min
    res$lambda.1se = parameters$lambda.1se
    
    # lambda.L <- res$lambda.min[1]
    # lambda.1 <- res$lambda.min[2]
    # lambda.2 <- res$lambda.min[3]
    
    lambda.L <- res$lambda.min[1]
    lambda.1 <- res$lambda.min[2]
    lambda.2 <- res$lambda.min[3]
  }
  # See Li & Li (2008) for reference
  Lnew <- lambda.L * L + lambda.2 * diag(p)
  eL <- eigen(Lnew)
  S <- eL$vectors %*% sqrt(diag(eL$values))
  l2star <- 1
  l1star <- lambda.1
  Xstar <- rbind(X, sqrt(l2star) * t(S)) / sqrt(1 + l2star)
  Ystar <- c(Y, rep(0, p))
  gammastar <- l1star / sqrt(1 + l2star) / 2 / (n + p)
  graceFit <- glmnet(Xstar, Ystar, lambda = gammastar, intercept = FALSE, standardize = FALSE, thresh = 1e-11)

  betahatstar <- graceFit$beta[, 1]
  betahat <- betahatstar / sqrt(1 + l2star)

  truebetahat <- betahat / scale.fac  # Scale back coefficient estimate
  truealphahat <- mean(ori.Y - as.matrix(ori.X) %*% truebetahat)
  
  res$fit = graceFit
  res$coefficients = truebetahat
  return(res)
}
