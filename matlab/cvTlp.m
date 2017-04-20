function [b_delta1, b_delta2, b_tau] = cvTlp(Y, X, wt, netwk, b0, ...
    deltas1, deltas2, taus, lasso_selection, k)
cv = cvpartition(size(Y,1),'k',double(k));

for i1 = 1:size(deltas1, 2)
    for i2 = 1:size(deltas2, 2)
        for i3 = 1:size(taus, 2)
            if lasso_selection == 0
                tau1 = taus(i3);
            else
                tau1 = 100;
            end
            
            fold_errors = zeros(k,1);
            for fold = 1:k
                train = cv.training(fold);
                holdout = cv.test(fold);
                b = tlp(Y(train,:),X(train,:),wt,netwk,b0,...
                    deltas1(i1),deltas2(i2),tau1,taus(i3));
                fold_errors(fold) = mean((X(holdout,:)*b - Y(holdout)).^2);
            end
            cur_mse = mean(fold_errors);

            fprintf( 'Delta 1 = %.2f,\t Delta 2 = %.2f,\t Tau = %.2f,\t MSE = %.2f\n', ...
                deltas1(i1), deltas2(i2), taus(i3), cur_mse)
            if exist('best_mse', 'var') == 0 || cur_mse < best_mse
                b_delta1 = deltas1(i1);
                b_delta2 = deltas2(i2);
                b_tau = taus(i3);
                best_mse = cur_mse;
            end
        end
    end
end

