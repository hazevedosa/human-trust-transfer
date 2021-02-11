close all; clear all; clc;

load('resultsRobotTrust_2Dim.mat')



norm_epochs = linspace(0, 1, length(tt));


figure(1)
set(gcf, 'Position', [10 10 3000 800])
subplot(1, 4, 1)

plot(norm_epochs, l_1, norm_epochs, u_1);
axis equal
axis([0 1 0 1])
set(gca,'XTick',[0:0.1:1])
grid on

subplot(1, 4, 2)
plot(norm_epochs, l_2, norm_epochs, u_2);
axis equal
axis([0 1 0 1])
set(gca,'XTick',[0:0.1:1])
grid on



stride = 200;
mm = mod(length(tt), stride);

rectangles = floor(length(tt) / stride);

idxs = zeros(1, rectangles);
for i = 1:rectangles
    idxs(i) = mm + i * stride;
end

areas = (u_1 - l_1) .* (u_2 - l_2);




subplot(1, 4, 3)
for i = 1:rectangles
    
    poly_Xs = [l_1(idxs(i)) u_1(idxs(i)) u_1(idxs(i)) l_1(idxs(i))];
    poly_Ys = [l_2(idxs(i)) l_2(idxs(i)) u_2(idxs(i)) u_2(idxs(i))];
    
    
    pgon = polyshape(poly_Xs, poly_Ys);
    plot(pgon, 'FaceColor', [1 1 1], 'FaceAlpha', 1.0)
    axis equal
    axis([0 1 0 1])
    set(gca,'XTick',[0:0.1:1])
    hold on
    
end

% plot(0.55, 0.75, '.g')


%-------------------------------------

load('fixed_tasks_robotTrust.mat');

subplot(1, 4, 4)
for i = 1:total_num_tasks
    if perfs_from_mat(i) == 1
        plot(p_from_mat(1, i), p_from_mat(2, i), '.b');
    else
        plot(p_from_mat(1, i), p_from_mat(2, i), '.r');
    end
    axis equal
    axis([0 1 0 1])
    set(gca,'XTick',[0:0.1:1])
    hold on
end


