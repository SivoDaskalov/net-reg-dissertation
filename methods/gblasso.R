#####################################################
# Network-based linear regression by GBLasso.
# Wei Pan, 3/27/08
#####################################################

batchGBLasso = function(Xtu, Ytu, Xtr, Ytr, Xts, Yts, edges, degrees, Betas){
  wt = degrees^(1/2)
  models = list()
  for(i in 1:nrow(Betas)){
    cat(timestamp("Setup "), i, "\n")
    # Currently training with the tuning datasets
    tuning <<- GBLasso(Xtu, Ytu[,i], netwk = edges, wt = wt)
    minMse = mse(Y = Yts[,i], X = Xts, b = tuning$betas[1,])
    solIdx = 1
    for(j in 2:tuning$nsol){
      curMse = mse(Y = Yts[,i], X = Xts, b = tuning$betas[j,])
      if(curMse < minMse){
        minMse = curMse
        solIdx = j
      }
    }
    tuning$coefficients = tuning$betas[solIdx,]
    model = list(fit = tuning, coefficients = tuning$betas[solIdx,])
    models[[i]] = model
  }
  return(batchEvaluateModels(Xts, Yts, models, Betas))
}

## squared error loss:
Xb<-function(x, b){
  sum(x*b)
}

LS<-function(Y,X,b){
  r<-Y-apply(X, 1, Xb, b)
  sum(r*r)
}

mse<- function(Y,X,b){
  return(LS(Y, X, b)/length(Y))
}

## network-based penalty value:
T<-function(b, netwk, gamma, wt){
  T0<-0
  K<-nrow(netwk)
  if (gamma>=1){
    for(k in 1:K)
      T0<-T0 + ( (abs(b[netwk[k,1]])^gamma)/wt[netwk[k,1]] +
                   (abs(b[netwk[k,2]])^gamma)/wt[netwk[k,2]] )^(1/gamma)
  } else{
    for(k in 1:K)
      T0<-T0 + max( abs(b[netwk[k,1]]/wt[netwk[k,1]]),  
                    abs(b[netwk[k,2]]/wt[netwk[k,2]])) 
  }
  T0
}

deltaT<-function(b, dbj, j, netwk, gamma, wt){
  dT0<-0
  K<-nrow(netwk)
  K1<-(1:K)[netwk[,1]==j]
  K2<-(1:K)[netwk[,2]==j]
  if (gamma>=1){
    for(k in K1){
      dT0<-dT0 + ( ((abs(b[netwk[k,1]]+dbj))^gamma)/wt[netwk[k,1]] +
                     (abs(b[netwk[k,2]])^gamma)/wt[netwk[k,2]] )^(1/gamma) -
        ( (abs(b[netwk[k,1]])^gamma)/wt[netwk[k,1]] +
            (abs(b[netwk[k,2]])^gamma)/wt[netwk[k,2]] )^(1/gamma)
    }
    for(k in K2){
      dT0<-dT0 + ( (abs(b[netwk[k,1]])^gamma)/wt[netwk[k,1]] +
                     ((abs(b[netwk[k,2]]+dbj))^gamma)/wt[netwk[k,2]] )^(1/gamma) -
        ( (abs(b[netwk[k,1]])^gamma)/wt[netwk[k,1]] +
            (abs(b[netwk[k,2]])^gamma)/wt[netwk[k,2]] )^(1/gamma)
    }
  } else{
    for(k in K1)
      dT0<-dT0 + max( abs((b[netwk[k,1]]+dbj)/wt[netwk[k,1]]),  
                      abs(b[netwk[k,2]]/wt[netwk[k,2]])) -
        max( abs(b[netwk[k,1]]/wt[netwk[k,1]]),
             abs(b[netwk[k,2]]/wt[netwk[k,2]]))
    for(k in K2)
      dT0<-dT0 + max( abs(b[netwk[k,1]]/wt[netwk[k,1]]),  
                      abs((b[netwk[k,2]]+dbj)/wt[netwk[k,2]])) -
        max( abs(b[netwk[k,1]]/wt[netwk[k,1]]),
             abs(b[netwk[k,2]]/wt[netwk[k,2]]))
  }
  dT0
}

##GBLasso for least squares loss and our network-based penalty:
##Input:
##       Y: response; X: covariates;
##       netwk: adjacency matrix based on a network; only two columns;
##       gamma: gamma-norm used in the penalty
##       wt: weight used for the norm;
GBLasso<-function(X, Y, netwk, gamma=2, wt, epsilon=0.1, ksi=0.01, MAXITER=1000){
  
  n<-length(Y);
  p<-length(X[1,]);
  
  b0<-rep(0, p)
  etaX<-rep(0, p)
  for(j in 1:p)
    etaX[j]<-sum(Y*X[,j])
  
  #step 1:
  b1<-b0
  while(T(b1, netwk, gamma, wt)==T(b0, netwk, gamma, wt)){
    b0<-b1
    jhat<-which.max(abs(etaX))
    if (length(jhat)>1) jhat<-jhat[1]
    shat<-sign(etaX[jhat])*epsilon
    b1[jhat]<-b0[jhat] + shat
    for(j in 1:p)
      etaX[j]<-etaX[j]- shat*sum(X[,jhat]*X[,j])
  }
  lambda0<-(LS(Y,X, b0) - LS(Y,X, b1))/
    (T(b1, netwk, gamma, wt) - T(b0, netwk, gamma, wt))
  
  betas<-matrix(0, nrow=MAXITER, ncol=p)
  lambdas<-rep(0,MAXITER) 
  
  nsol<-1
  betas[nsol,]<-b0
  lambdas[nsol]<-lambda0
  
  #steps 2-3:
  lambda1<-lambda0
  while (lambda1>0 && nsol<MAXITER){
    b0<-b1
    lambda0<-lambda1
    
    s<-epsilon
    dT<-rep(0, p)
    for(j in 1:p){
      b01<-b0
      b01[j]<-b01[j]+s
      dT[j]<-deltaT(b0, s, j, netwk, gamma, wt) 
      #cat("j=", j, " deltaT=", dT[j], " but dT=", dT[j]<-T(b01, netwk, gamma, wt) - T(b0, netwk, gamma, wt), "\n")
    }
    dGamma1<-0-2*s*etaX + s*s*(n-1) + lambda0*dT
    
    s<-0-epsilon
    dT<-rep(0, p)
    b01<-b0
    for(j in 1:p){
      b01<-b0
      b01[j]<-b01[j]+s
      dT[j]<-deltaT(b0, s, j, netwk, gamma, wt) 
      #cat("j=", j," deltaT=", dT[j], " But dT=", dT[j]<-T(b01, netwk, gamma, wt) - T(b0, netwk, gamma, wt), "\n")
    }
    dGamma2<-0-2*s*etaX + s*s*(n-1) + lambda0*dT
    
    j1hat<-which.min(dGamma1)
    j2hat<-which.min(dGamma2)
    if (dGamma1[j1hat]<=dGamma2[j2hat]) {
      jhat<-j1hat
      dGamma<-dGamma1[j1hat]
      shat<-epsilon
    } else {
      jhat<-j2hat
      dGamma<-dGamma2[j2hat]
      shat<-0-epsilon
    }
    
    if (dGamma < 0-ksi){
      b1<-b0
      b1[jhat]<-b1[jhat] + shat
      lambda1<-lambda0
      for(j in 1:p)
        etaX[j]<-etaX[j]- shat*sum(X[,jhat]*X[,j])
      #cat("F: ", jhat, shat, dGamma, lambda0,"\n")
      
    } else{
      #record current sol:
      if (lambda0<lambdas[nsol]){
        nsol<-nsol+1
        betas[nsol,]<-b0
        lambdas[nsol]<-lambda0
      }
      
      jhat<-which.max(abs(etaX))
      if (length(jhat)>1) jhat<-jhat[1]
      # below could result a dead loop b/w F<->B; e.g. when sim=3, wt=1
      #shat<-sign(etaX[jhat])*epsilon
      # thus modify the step size so that F & B will NOT overlap:
      shat<-sign(etaX[jhat])*epsilon*sqrt(2)
      b1<-b0
      b1[jhat]<-b0[jhat] + shat
      for(j in 1:p)
        etaX[j]<-etaX[j]- shat*sum(X[,jhat]*X[,j])
      
      lambda0<-(LS(Y,X, b0) - LS(Y,X, b1))/
        (T(b1, netwk, gamma, wt) - T(b0, netwk, gamma, wt))
      lambda1<-min(lambda1, lambda0)
      #cat("B: ", jhat, shat, lambda0, lambda1, "\n")
    }
  } #while
  
  return(structure(list(betas=betas[1:nsol,], lambdas=lambdas[1:nsol], nsol=nsol), class = "GBLassoFit"))
}

predict.GBLassoFit = function(object, data, type = "response") {
  return(apply(data*object$coefficients,1,sum))
} 
