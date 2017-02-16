source("matlab_cvx_tools.R")

n <- 50
p <- 20
k <- 10

x <- matrix(rnorm(n * p), n, p)
y <- x %*% rnorm(p) + 0.1 * rnorm(n)

tuning.params = list(gam = c(1,2), lam = c(4,5), wut = c(0.5, 3.5))
const.vars = list(k = k, n = n, p = p)

cvxcode <- paste("variables b(p)",
                 "minimize(square_pos(norm(y - x * b, 2)) / 2 + lam * norm(b, 1) / (gam + wut));",
                 sep="; ")

cvxResults = cvConvexOptim(x = x, y = y, cvxcode = cvxcode, tuning.params = tuning.params, const.vars = const.vars, k = 10, title = "example")
errors = cvxResults$results
