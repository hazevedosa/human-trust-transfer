% just thinking...

close all; clear all; clc;

A = readmatrix('Driving_reqs.csv');

A = A(:, 18:end);

for i=1:size(A,1)
    for j=1:size(A,2)
        if isnan(A(i, j))
            A(i,j) = 0;
        end
    end    
end

Long = [];

for task = 1:12
    M_task = [A(:, 3 * (task - 1) + 1), A(:, 3 * (task - 1) + 2), A(:, 3 * (task - 1) + 3)];
    Long = [Long; M_task];
end

mu = mean(Long)
C = cov(Long)


CC = (Long - mu)' * (Long - mu)/371;


task = 12; % 1 to 12
% 
% M_task = [A(:, 3 * (task - 1) + 1), A(:, 3 * (task - 1) + 2), A(:, 3 * (task - 1) + 3)];
% 
% for i = 1:size(M_task, 1)
%     plot3(M_task(i, 1), M_task(i, 2), M_task(i, 3), '.');
%     hold on
% end
% 
% plot3(68, 71, 56,'*')
% 
% xlabel('Cognitive')
% ylabel('Sensory')
% zlabel('Physical')
% grid
% axis([0 100 0 100 0 100])
% 
capab = 3; % 1-Cognitive; 2-Sensory; 3-Physical

col = 3 * (task - 1) + capab;

histogram(A(:, col), 100)
