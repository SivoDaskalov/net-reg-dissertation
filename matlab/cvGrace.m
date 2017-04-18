function [b_lam1, b_lam2] = cvGrace(Y, X, wt, netwk, a, lam1, lam2, k)
cv = cvpartition(size(Y,1),'k',double(k));

for i1 = 1:size(lam1, 2)
    for i2 = 1:size(lam2, 2)
        fold_errors = zeros(k,1);
        for fold = 1:k
            train = cv.training(fold);
            holdout = cv.test(fold);
            b = grace(Y(train,:),X(train,:),wt,netwk,a,lam1(i1),lam2(i2));
            fold_errors(fold) = mean((X(holdout,:)*b - Y(holdout)).^2);
        end
        cur_mse = mean(fold_errors);
        
        fprintf( 'Lambda 1 = %.2f,\t Lambda 2 = %.2f,\t MSE = %.2f\n', ...
            lam1(i1),lam2(i2), cur_mse)
        if exist('best_mse', 'var') == 0 || cur_mse < best_mse
            b_lam1 = lam1(i1);
            b_lam2 = lam2(i2);
            best_mse = cur_mse;
        end
    end;
end;