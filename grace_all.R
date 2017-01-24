# codes modified from package "Grace"

library(glmnet)

cvGrace <- function(X, Y, L, lambda.L, lambda.1, lambda.2, K = 10){
    lambda.1 <- unique(sort(lambda.1, decreasing = TRUE))
    lambda.L <- unique(sort(lambda.L, decreasing = TRUE))
    lambda.2 <- unique(sort(lambda.2, decreasing = TRUE))

    p <- ncol(X)
    n <- nrow(X)
    ERRlist = c()
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
            for( i1 in 1:length(gammastar)    ){
                temp_gammastar =  c(gammastar[i1] + 0.01, gammastar[i1])
                cvres <- cv.glmnet(Xstar, Ystar, lambda = temp_gammastar, intercept = FALSE, standardize = FALSE, nfolds = K)
                if(!is.na(cvres$cvm[2])){
                    ERRlist = rbind( ERRlist, c( lambda.1[i1] , lL, l2, cvres$cvm[2] )  )
                }
            }
        }
    }
    
    colnames(ERRlist)=c("lambda.1","lambda.L","lambda.2","cvm")
    return(ERRlist)
}

grace <- function(Y, X, L, lambda.L, lambda.1 = 0, lambda.2 = 0, normalize.L = FALSE, K = 10){
  lambda.L <- unique(sort(lambda.L, decreasing = TRUE))
  lambda.1 <- unique(sort(lambda.1, decreasing = TRUE))
  lambda.2 <- unique(sort(lambda.2, decreasing = TRUE))
  
  ori.Y <- Y
  ori.X <- X
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
  
  Y <- Y - mean(Y)  # Center Y
  n <- nrow(X)
  p <- ncol(X)
  scale.fac <- attr(scale(X), "scaled:scale")
  X <- scale(X)     # Standardize X
  
  if(normalize.L){
    diag(L)[diag(L) == 0] <- 1
    L <- diag(1 / sqrt(diag(L))) %*% L %*% diag(1 / sqrt(diag(L)))  # Normalize L
  }
  
  # If more than one tuning parameter is provided, perform K-fold cross-validation  
  if((length(lambda.L) > 1) | (length(lambda.1) > 1) | (length(lambda.2) > 1)){
    parameters <- cvGrace(X, Y, L, lambda.L, lambda.1, lambda.2, K)
    tun = parameters[parameters[,4]==min(parameters[,4]), ]
    
    if(length(dim(tun))==2 ){
      lambda.1 <- tun[1,1]
      lambda.L <- tun[1,2]
      lambda.2 <- tun[1,3]  
    }else{
      lambda.1 <- tun[1]
      lambda.L <- tun[2]
      lambda.2 <- tun[3] 
    }
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
  truealphahat <- mean(ori.Y - ori.X %*% truebetahat)
  return(list( ParameterEstimation = list(parameterCV=parameters, parameterMin=tun),
               GraceFit = graceFit,
               Beta=list(intercept = truealphahat, beta = truebetahat)))
}

grace(Y,X,L)