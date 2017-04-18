function best_e = cvAlinf(Y, X, wt, netwk, a, d, k, E)
cv = cvpartition(size(Y,1),'k',double(k));
for i=1:size(E,2)
    fold_errors = zeros(k,1);
    for fold = 1:k
        train = cv.training(fold);
        holdout = cv.test(fold);
        b = alinf(Y(train,:), X(train,:), wt, netwk, a, d, E(i));
        fold_errors(fold) = mean((X(holdout,:)*b - Y(holdout)).^2);
    end
    cur_mse = mean(fold_errors);
    
    if exist('best_mse', 'var') == 0 || cur_mse < best_mse
        best_e = E(i);
        best_mse = cur_mse;
    end
end


