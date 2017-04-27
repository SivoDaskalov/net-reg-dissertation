function coef = leastsquares(Y, X)
p = size(X,2);
cvx_begin quiet
%   cvx_precision high;
  variable b(p);
  minimize(norm(Y-X*b));
cvx_end
coef = b;
