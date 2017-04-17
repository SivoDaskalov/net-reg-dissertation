function [coef,lam,gam,mse] = cvGblasso(Y, X, wt, netwk, lam_all, gam_all, k)
cv = cvpartition(size(Y,1),'k',k);
for lam_idx = 1:size(lam_all, 2)
    clam=lam_all(1,lam_idx);
    for gam_idx = 1:size(gam_all, 2)
        cgam=gam_all(1,gam_idx);
        
        fold_errors = zeros(k,1);
        for fold = 1:k
            train = cv.training(fold);
            holdout = cv.test(fold);
            b = gblasso(Y(train,:), X(train,:), wt, netwk, clam, cgam);
            fold_errors(fold) = mean((X(holdout,:)*b - Y(holdout)).^2);
        end
        cur_mse = mean(fold_errors);
        
        if exist('mse', 'var') == 0 || cur_mse < mse
            coef = b;
            lam = clam;
            gam = cgam;
            mse = cur_mse;
        end
    end;
end;