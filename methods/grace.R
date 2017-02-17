
grace <- function(xtr, ytr, xtu, ytu, network, degrees, lambda.1 = 0, lambda.2 = 0, adjustments, k = 10, norun = FALSE){
  
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
  
  return(cvxTuneAndTrain(xtr = xtr, ytr = ytr, xtu = xtu, ytu = ytu, cvxcode = graceConvexCode, tuning.params = tuning.params, const.vars = const.vars, title = title, norun = FALSE))
}
