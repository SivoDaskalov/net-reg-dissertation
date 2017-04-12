%% calculate the ``network norm''
% norm = sum_{i~j} norm([|x_i|/w_i, |x_j|/w_j], gamma)
% here we assume: 
% x: a column vector of size px1
% wt: a column vector of size px1   
% netwk: a matrix of size mx2
% gamma >= 1 is a constant (e.g. 1, 2, 8, Inf) 
 
function y = netwknorm(x, netwk, wt, gamma)
tmp = [x(netwk(:,1))./wt(netwk(:,1)) x(netwk(:,2))./wt(netwk(:,2))];
y = sum(norms(tmp, gamma, 2));

