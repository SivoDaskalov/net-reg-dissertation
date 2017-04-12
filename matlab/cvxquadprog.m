function [y,exitflag] = cvxquadprog(Y, X, netwk, wt, gamma, C)
p = size(X,2);
cvx_begin quiet
  cvx_precision high;
  variable x(p);
  minimize(norm(Y-X*x));
  subject to
    netwknorm(x, netwk, wt, gamma) <= C;
cvx_end
exitflag = cvx_status;
y = x;

