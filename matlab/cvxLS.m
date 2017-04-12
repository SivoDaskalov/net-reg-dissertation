function [y,exitflag] = cvxLS(Y, X)
p = size(X,2);
cvx_begin quiet
  %cvx_precision high;
  variable x(p);
  minimize(sum_square(Y-X*x));
cvx_end
exitflag = cvx_status;
y = x;

