function coef = alinf(Y, X, wt, netwk, a, d, E)
p = size(X,2);
cvx_begin quiet
%   cvx_precision high;
  cvx_solver sdpt3;
  variable b(p);
  minimize(norm(Y-X*b));
  subject to
    sum(abs(b(netwk(:,1))./wt(netwk(:,1))- ...
        a(:).*b(netwk(:,2))./wt(netwk(:,2)))) <= E;
    b(d(:)) == 0;
cvx_end
coef = b;
