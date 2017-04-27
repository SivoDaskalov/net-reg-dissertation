function netnorm = infnorm(x, wt, netwk)
c1 = x(netwk(:,1))./wt(netwk(:,1));
c2 = x(netwk(:,2))./wt(netwk(:,2));
% netnorm = sum(norms([c1, c2], Inf, 2));
netnorm = 0;
for i = 1:size(netwk,1)
    netnorm = netnorm + max(abs(c1(i)),abs(c2(i)));
end;
