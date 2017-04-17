function [lam1,lam2] = cvGrace(Y, X, wt, netwk, a, lam1_all, lam2_all, k)
cv = cvpartition(size(Y,1),'k',k);
for lam1_idx = 1:size(lam1_all, 2)
    clam1=lam1_all(1,lam1_idx);
    for lam2_idx = 1:size(lam2_all, 2)
        clam2=lam2_all(1,lam2_idx);
        
        fold_errors = zeros(k,1);
        for fold = 1:k
            train = cv.training(fold);
            holdout = cv.test(fold);
            b = grace(Y(train,:), X(train,:), wt, netwk, a, clam1, clam2);
            fold_errors(fold) = mean((X(holdout,:)*b - Y(holdout)).^2);
        end
        cur_mse = mean(fold_errors);
        
        if exist('mse', 'var') == 0 || cur_mse < mse
            lam1 = clam1;
            lam2 = clam2;
            mse = cur_mse;
        end
    end;
end;