library(glmnet)
library(parallel)

convexGraceTuning = function(X, Y, network, degrees, lambda.1, lambda.2, weights = NULL, adjustments = NULL, k = 10, norun = FALSE){
  
  # --- General Adjusted Grace Model is as follows ---
  # variable b(p)
  # pen = b(netwk(:,1))./deg(netwk(:,1))-a(:).*b(netwk(:,2))./deg(netwk(:,2))
  # minimize(sum_square(y-x*b)+lam1*norm(b,1)+lam2*sum((pen.^2).*wt(:)));
  
  title = "Adjusted Grace"
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
  
  return(cvConvexOptim(x = X, y = Y, cvxcode = graceConvexCode, tuning.params = tuning.params, const.vars = const.vars, title = title, norun = norun))
}

grace <- function(Xtr, Ytr, Xtu, Ytu, network, degrees, adjustments, lambda.1 = 0, lambda.2 = 0, k = 10, norun = FALSE){
  res = list()
  
  # If more than one tuning parameter is provided, perform K-fold cross-validation  
  if((length(lambda.1) > 1) || (length(lambda.2) > 1)){
    res$tuning = convexGraceTuning(X = Xtu, Y = Ytu, network = network, degrees = degrees, lambda.1 = lambda.1, 
                                   lambda.2 = lambda.2, adjustments = adjustments, k = k, norun = norun)
    if(norun == TRUE){
      return(res)
    }
    res$lambda.min = res$tuning$results[which.min(res$tuning$results[,ncol(res$tuning$results)]),]
    lambda.1 <- res$lambda.min[[1]]
    lambda.2 <- res$lambda.min[[2]]
  }
  
  # Integrate training with tuning in Matlab?
  return(res)
}

# grace <- function(Xtr, Ytr, Xtu, Ytu, network, degrees, lambda.1 = 0, lambda.2 = 0, K = 10){
#   res = list()
#   ori.Y <- Y
#   ori.X <- X
#   scale.fac <- attr(scale(X), "scaled:scale")
#   X <- scale(X)     # Standardize X
#   Y <- Y - mean(Y)  # Center Y
#   n <- nrow(X)
#   p <- ncol(X)
#   
#   if(missing(enetFit)){
#     # Assuming regular grace
#     b0 = rep(1, p)
#   }else{
#     # Assuming adaptive grace
#     if(p<n){
#       # Take initial estimates based on OLSE
#       lmData = data.frame(Y = Y, Xtu)
#       lmFit = lm(Y ~ .-1, data = Xtu)
#       b0 = lmFit$coefficients
#     } else{
#       # Take initial estimates based on ENet
#       b0 = enetFit$coefficients
#     }
#   }
#   res$b0 = b0
#   
#   # See Li & Li (2008) for reference
#   Lnew <- lambda.L * L + lambda.2 * diag(p)
#   eL <- eigen(Lnew)
#   S <- eL$vectors %*% sqrt(diag(eL$values))
#   l2star <- 1
#   l1star <- lambda.1
#   Xstar <- rbind(X, sqrt(l2star) * t(S)) / sqrt(1 + l2star)
#   Ystar <- c(Y, rep(0, p))
#   gammastar <- l1star / sqrt(1 + l2star) / 2 / (n + p)
#   graceFit <- glmnet(Xstar, Ystar, lambda = gammastar, intercept = FALSE, standardize = FALSE, thresh = 1e-11)
# 
#   betahatstar <- graceFit$beta[, 1]
#   betahat <- betahatstar / sqrt(1 + l2star)
# 
#   truebetahat <- betahat / scale.fac  # Scale back coefficient estimate
#   truealphahat <- mean(ori.Y - as.matrix(ori.X) %*% truebetahat)
#   
#   res$fit = graceFit
#   res$coefficients = truebetahat
#   return(res)
# }
