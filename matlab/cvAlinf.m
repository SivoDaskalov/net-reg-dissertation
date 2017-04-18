function best_e = cvAlinf(Y, X, wt, netwk, a, d, e, k)
cv = cvpartition(size(Y,1),'k',double(k));

for i=1:size(e,2)
    fold_errors = zeros(k,1);
    for fold = 1:k
        train = cv.training(fold);
        holdout = cv.test(fold);
        b = alinf(Y(train,:), X(train,:), wt, netwk, a, d, e(i));
        fold_errors(fold) = mean((X(holdout,:)*b - Y(holdout)).^2);
    end
    cur_mse = mean(fold_errors);
    
    fprintf( 'E = %.2f,\t MSE = %.2f\n', e(i), cur_mse)
    if exist('best_mse', 'var') == 0 || cur_mse < best_mse
        best_e = e(i);
        best_mse = cur_mse;
    end
end


