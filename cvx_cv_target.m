p = dlmread('temp_in_p.txt');
y = dlmread('temp_in_y.txt');
x = dlmread('temp_in_x.txt');
folds = dlmread('temp_in_folds.txt');
lam_all = dlmread('temp_in_lam_all.txt');
gam_all = dlmread('temp_in_gam_all.txt');

results = zeros(size(y),2+1); % 2 = number of tuned parameters

cd C:\Users\sivak_000\Documents\GitHub\dissertation\cvx;
cvx_setup;
cd('C:\Users\sivak_000\Documents\GitHub\dissertation');

tStart=tic;
k = max(folds);

counter = 1;
for lam_it=1:size(lam_all);
    lam=lam_all(lam_it);
    for gam_it=1:size(gam_all);
        gam=gam_all(gam_it);
        cvMse = zeros(k,1);
        for fold = 1:k;
            training = find(folds ~= fold);
            holdout = find(folds == fold);
            cvx_begin quiet;
            variables b(p);
            minimize(square_pos(norm(y - x*b, 2)) / 2 + lam*norm(b, 1)*gam);
            cvx_end;
            cvMse(fold) = mean((x(holdout,:)*b - y(holdout)).^2);
        end;
        mse = mean(cvMse);
        results(counter,:) = [lam,gam,mse];
        counter = counter + 1;
    end;
end;
time=toc(tStart);

dlmwrite('temp_out_results.txt', results, 'precision', '%10.10f');
dlmwrite('temp_out_time.txt', full(time), 'precision', '%10.10f');

% exit;
