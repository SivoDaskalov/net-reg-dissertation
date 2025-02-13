function best_c = cvLinf(Y, X, wt, netwk, c, k)
cv = cvpartition(size(Y,1),'k',double(k));

for i=1:size(c,2)
    fold_errors = zeros(k,1);
    for fold = 1:k
        train = cv.training(fold);
        holdout = cv.test(fold);
        b = linf(Y(train,:), X(train,:), wt, netwk, c(i));
        fold_errors(fold) = mean((X(holdout,:)*b - Y(holdout)).^2);
    end
    cur_mse = mean(fold_errors);
    
    fprintf( 'C = %.2f,\t MSE = %.2f\n', c(i), cur_mse)
    if exist('best_mse', 'var') == 0 || isnan(best_mse) || cur_mse < best_mse
        best_c = c(i);
        best_mse = cur_mse;
    end
end
