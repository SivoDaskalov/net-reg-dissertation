# %% Linf_w 
# %fit the original model
# gamma=Inf;
# [x,flag] = cvxquadprog(Y,X,netwk,wt,gamma,100); % the solution with virtually no constraint 
# maxC = ceil(netwknorm(x,netwk,wt,gamma));
# %C = 0.5:0.5:maxC;
# C = 1:1:maxC;
# k = size(C,2);
# 
# [x,flag] = cvxquadprog(Y, X, netwk, wt, gamma, C(1));
# minPSE0 = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
# x0 = x;
# C0 = C(1);
# for i=2:k,
# [x,flag] = cvxquadprog(Y, X, netwk, wt, gamma, C(i));
# PSEtu = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
# if(PSEtu < minPSE0)
#   minPSE0 = PSEtu;
# x0 = x;
# C0 = C(i);
# end
# end
# bLinf=x0;

linfw <- function(xtr, ytr, xtu, ytu, network, degrees, lambda.1 = 0, lambda.2 = 0, adjustments = NULL, weights = NULL, k = 10, norun = FALSE){
}

batchLinfw = function(xtu, ytu, xtr, ytr, xts, yts, network, degrees, lambda.1, lambda.2, betas, k=10){
}