function [y,exitflag] = cvxEnet(Y, X, lam1, lam2)
p = size(X,2);
cvx_begin quiet
  %cvx_precision high;
  variable x(p);
  minimize(sum_square(Y-X*x)+lam1*norm(x,1)+lam2*sum_square(x));
cvx_end
exitflag = cvx_status;
y = x;

