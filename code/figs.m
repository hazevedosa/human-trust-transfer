close all; clear all; clc;

results_gp  = load('results_mat_gp.mat');
results_btm = load('results_mat_btm.mat');
results_opt = load('results_mat_lineargaussian.mat');

lcs_gp  = results_gp.allresults(:, 4);
lcs_btm = results_btm.results(:, 4);
lcs_opt = results_opt.allresults(:, 4);

y_min = 0.5;
y_max = 0.9;


m = 1;

for i=1:size(lcs_gp, 1)
    m = max(m, size(lcs_gp{i}, 2));
end


data = zeros(3, m, 10);
data_avg_gp = zeros(3, m);

 
for i=1:size(lcs_gp, 1)
    data_ = lcs_gp{i};
    data(1, 1:m, i) = 1:m;
    data(2:end, 1:size(data_, 2), i) = data_(2:end, :);
    data(2, size(data_, 2):end, i) = data_(2, end);
    data(3, size(data_, 2):end, i) = data_(3, end);
    data_avg_gp = data_avg_gp + data(:, :, i)/10;
end

sdevs_gp = zeros(2, m);

for i=1:2
    for j = 1:m
        sdevs_gp(i, j) = std(data(i+1, j, :));
    end
end

curve = plot(data_avg_gp(1, :), data_avg_gp(2, :), 'LineWidth', 1)
hold on
% plot(data_avg_gp(1, :), data_avg_gp(2, :) + sdevs_gp(1, :));
% plot(data_avg_gp(1, :), data_avg_gp(2, :) - sdevs_gp(1, :));

poly_Xs = [data_avg_gp(1, :), flip(data_avg_gp(1, :))];
poly_Ys = [data_avg_gp(2, :) - sdevs_gp(1, :), flip(data_avg_gp(2, :) + sdevs_gp(1, :))];

pgon = polyshape(poly_Xs, poly_Ys);
shadows = plot(pgon, 'FaceColor','black','FaceAlpha',0.05);
shadows.LineStyle = 'none';
% shadows.EdgeColor = 'red';
axis([1 m y_min y_max])

% figure(1)
% subplot(1, 3, 1)
% plot(data_avg_gp(1, :), data_avg_gp(2, :))
% axis([0 size(data_avg_gp(1, :), 2) 0.59 0.85])
% 
% subplot(1, 3, 2)
% plot(data_avg_btm(1, :), data_avg_btm(2, :))
% axis([0 size(data_avg_btm(1, :), 2) 0.59 0.85])
% 
% subplot(1, 3, 3)
% plot(data_avg_opt(1, :), data_avg_opt(2, :))
% axis([0 size(data_avg_opt(1, :), 2) 0.59 0.85])
