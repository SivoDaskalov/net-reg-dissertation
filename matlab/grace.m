function coef = grace(Y, X, wt, netwk, a, lam1, lam2)
p = size(X,2);
cvx_begin quiet
  variable b(p);
  tmp = b(netwk(:,1))./wt(netwk(:,1))-a(:).*b(netwk(:,2))./wt(netwk(:,2));
  minimize(sum_square(Y-X*b)+lam1*norm(b,1)+lam2*sum_square(tmp));
cvx_end
coef = b;