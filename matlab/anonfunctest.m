function y = anonfunctest()

netnorm = @(x, netwk, wt, gamma) sum(norms([x(netwk(:,1))./wt(netwk(:,1)), x(netwk(:,2))./wt(netwk(:,2))], gamma, 2));
wt = [1;1;1;1;1;1];
netwk = [1,2;1,3;4,5;4,6];
x = [6;5;4;3;2;1];
gamma = 2;
y = netnorm(x, netwk, wt, gamma);

end

