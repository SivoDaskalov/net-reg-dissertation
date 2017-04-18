function netnorm = infnorm(x, wt, netwk)
netnorm = sum(norms([x(netwk(:,1))./wt(netwk(:,1)), x(netwk(:,2))./wt(netwk(:,2))], Inf, 2));
