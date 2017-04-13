
b = c(6, 5, 4, 3, 2, 1)
edges = Matrix(data = c(1, 2, 1, 3, 4, 5, 4, 6), nrow=4, ncol=2, byrow = TRUE)
wt = c(1, 1, 1, 1, 1, 1)

# x: a vector of length p
# wt: a vector of length p 
# netwk: a matrix of shape(m, 2)
# gamma >= 1 is an integer constant or Inf 
networkNorm = function(b, edges, wt, gamma) {
  tmp = cbind(b[edges[, 1]] / wt[edges[, 1]], b[edges[, 2]] / wt[edges[, 2]])
  if (is.infinite(gamma)) {
    return(sum(apply(tmp, 1, function(x)
      norm(as.matrix(x), type = "I"))))
  }
  return(sum(apply(tmp, 1, function(x)
    sum(abs(x) ^ gamma) ^ (1 / gamma))))
}

networkNorm(b, edges, wt, 2)
