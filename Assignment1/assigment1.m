%% TASK 4

clear
close all
load A1_data.mat

lambda1 = 0.1;
omega_hat1 = lasso_ccd(t, X, lambda1);
y_hat1 = Xinterp * omega_hat1;
y_data1 = X * omega_hat1;

lambda2 = 10;
omega_hat2 = lasso_ccd(t, X, lambda2);
y_hat2 = Xinterp * omega_hat2;
y_data2 = X * omega_hat2;

lambda3 = 1.5;
omega_hat3 = lasso_ccd(t, X, lambda3);
y_hat3 = Xinterp * omega_hat3;
y_data3 = X * omega_hat3;

figure
hold on
scatter(n, t, 30, 'b')
scatter(n, y_data1, 30, 'filled')
plot(ninterp, y_hat1, 'r')
legend('Real Data Points', 'Synthesized Data Points', 'Reconstruction')
xlabel('Time')
text(1,6,strcat('\lambda = ',sprintf('%.1f',lambda1)))

figure
hold on
scatter(n, t, 30, 'b')
scatter(n, y_data2, 30, 'filled')
plot(ninterp, y_hat2, 'r')
legend('Real Data Points', 'Synthesized Data Points', 'Reconstruction')
xlabel('Time')
text(1,6,strcat('\lambda = ',sprintf('%.1f',lambda2)))

figure
hold on
scatter(n, t, 30, 'b')
scatter(n, y_data3, 30, 'filled')
plot(ninterp, y_hat3, 'r')
legend('Real Data Points', 'Synthesized Data Points', 'Reconstruction')
xlabel('Time')
text(1,6,strcat('\lambda = ',sprintf('%.1f',lambda3)))

non_zero_coordinates1 = sum(omega_hat1~=0); % 253
non_zero_coordinates2 = sum(omega_hat2~=0); % 8
non_zero_coordinates3 = sum(omega_hat3~=0); % 61

%% Task 5
clear;
close all
load A1_data.mat

lambda_min = 0.01;
lambda_max = max(abs(X'*t));
N_lambda = 100;
N_folds = 10;

lambda_grid = exp( linspace( log(lambda_min), log(lambda_max), N_lambda));
[wopt, lambdaopt, RMSEval, RMSEest] = lasso_cv(t, X, lambda_grid, N_folds);

t_optimal = Xinterp*wopt;
t_data = X*wopt;
%%
legends = [];
figure
hold on
legends = [legends scatter(log(lambda_grid), RMSEval, 'bx')];
plot(log(lambda_grid), RMSEval, 'b')
legends = [legends scatter(log(lambda_grid), RMSEest, 'cx')];
plot(log(lambda_grid), RMSEest, 'c')
legends = [legends xline(log(lambdaopt), '--r')];
legend(legends, 'Validation RMSE', 'Estimate RMSE', 'Optimal \lambda')
xlabel('log(\lambda)')
txt = strcat('\leftarrow \lambda = ', sprintf('%.1f',lambdaopt));
text(log(lambdaopt),15,txt)

omega_hat_opt = lasso_ccd(t, X, lambdaopt);
y_hat_opt = Xinterp * omega_hat_opt;
y_data_opt = X * omega_hat_opt;

figure
hold on
scatter(n, t, 30, 'b')
scatter(n, y_data_opt, 30, 'filled')
plot(ninterp, y_hat_opt, 'r')
legend('Real Data Points', 'Synthesized Data Points', 'Reconstruction')
xlabel('Time')
text(1,6,strcat('\lambda = ',sprintf('%.1f',lambdaopt)))

%% TASK 6 DO NOT RUN UNLESS YOU NEED TO RECALCULATE
clear;
close all
load A1_data.mat

lambda_min = 0.0005;
N_lambda = 100;
N_folds = 3;
frame_length = size(Xaudio, 1);
N_frames = floor(length(Ttrain)./frame_length);
lambda_maxes = zeros(N_frames, 1);

for frame_index=1:N_frames
    lambda_maxes(frame_index) = max(abs(Xaudio'*Ttrain(1 + frame_length*(frame_index-1) : frame_index*frame_length)));
end

lambda_max = max(lambda_maxes);
lambda_grid = exp( linspace( log(lambda_min), log(lambda_max), N_lambda));

[wopt, lambdaopt, RMSEval, RMSEest] = multiframe_lasso_cv(Ttrain, Xaudio, lambda_grid, N_folds);

%% PLOTS
load A1_data.mat
load multiframe_results.mat
legends = [];

figure
hold on
legends = [legends scatter(log(lambda_grid), RMSEval, 'bx')];
plot(log(lambda_grid), RMSEval, 'b')
legends = [legends scatter(log(lambda_grid), RMSEest, 'cx')];
plot(log(lambda_grid), RMSEest, 'c')
legends = [legends xline(log(lambdaopt), '--r')];
legend(legends, 'Validation RMSE', 'Estimate RMSE', 'Optimal \lambda')
xlabel('log(\lambda)')
txt = strcat('\leftarrow \lambda = ', sprintf('%.3f',lambdaopt));
text(log(lambdaopt),0.25,txt)

%% TASK 7
clear;
load A1_data.mat
load multiframe_results.mat

Ytest = lasso_denoise(Ttest, Xaudio, lambdaopt);
soundsc(Ytest, fs);

save('denoised_audio','Ytest','fs');











