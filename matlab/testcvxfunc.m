function coef = testcvxfunc(X, y)
p = size(X,2);
cvx_begin quiet
    variable b(p)
    minimize( norm(X*b - y) )
cvx_end
coef = b;
end

