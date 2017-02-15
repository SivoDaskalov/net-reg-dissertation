n <- 50
p <- 10
x <- matrix(rnorm(n * p), n, p)
beta <- rnorm(p)
y <- x %*% beta + 0.1 * rnorm(n)
library(CVXfromR)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setup.dir <- "cvx"
cvxcode <- paste("variables b(p)",
                 "minimize(square_pos(norm(y - x * b, 2)) / 2 + lam * norm(b, 1))",
                 sep=";")
lasso <- CallCVX(cvxcode, const.vars=list(p=p, y=y, x=x, lam=2),
                 opt.var.names="b", setup.dir=setup.dir)
names(lasso)

# lasso <- CallCVX.varyparam(cvxcode, const.vars=list(p=p, y=y, x=x), tuning.param=list(lam=seq(0.1, 1, length=20)),
#                            opt.var.names="b", setup.dir=setup.dir)