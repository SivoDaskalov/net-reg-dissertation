function coef = gblasso(Y, X, wt, netwk, lam, gam)
p = size(X,2);
pen_mult = lam * 2^(1-(1/gam));
cvx_begin quiet
    %cvx_precision high;
    variable b(p);
    penalty = ...
        pow_p(b(netwk(:,1)),gam)./wt(netwk(:,1))+...
        pow_p(b(netwk(:,2)),gam)./wt(netwk(:,2));
    minimize(sum_square(Y-X*b)+pen_mult*sum(pow_p(penalty),1/gam));
cvx_end
coef = b;