function netnorm = infnorm(x, wt, netwk)
c1 = abs(x(netwk(:,1))./wt(netwk(:,1)));
c2 = abs(x(netwk(:,2))./wt(netwk(:,2)));
netnorm = sum(max(c1,c2));
% netnorm = sum(norms([c1, c2], Inf, 2));