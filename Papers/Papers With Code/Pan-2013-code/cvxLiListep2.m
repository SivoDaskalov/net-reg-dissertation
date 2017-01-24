function [y,exitflag] = cvxLiListep2(Y, X, wt, netwk, a, lam1, lam2)
p = size(X,2);
cvx_begin quiet
  %cvx_precision high;
  variable x(p);
  tmp = x(netwk(:,1))./wt(netwk(:,1))-a(:).*x(netwk(:,2))./wt(netwk(:,2));
  minimize(sum_square(Y-X*x)+lam1*norm(x,1)+lam2*sum_square(tmp));
cvx_end
exitflag = cvx_status;
y = x;

