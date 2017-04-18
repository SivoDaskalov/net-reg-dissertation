function coef = linf(Y, X, wt, netwk, C)
p = size(X,2);
cvx_begin quiet
  cvx_precision high;
  cvx_solver sdpt3;
  variable b(p);
  minimize(norm(Y-X*b));
  subject to
    infnorm(b, wt, netwk) <= C;
cvx_end
coef = b;
