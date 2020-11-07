% just thinking...

close all; clear all; clc;



P = [10 0; 0 10];

C = [1 0; 0 1];

Q = [1 0; 0 1];


figure(1)
% plot_ellip([0 0], P, 1)
plot(0, 0, '*b')
axis([-5 5 -5 5])
axis equal
grid
hold on

pause(1)

P = P - P * C' * inv(C * P * C' + Q) * C * P

% plot_ellip([1 2], P, 1)
plot(1, 2, '*b')


pause(1)
P = P - P * C' * inv(C * P * C' + Q) * C * P
% plot_ellip([2 1], P, 1)
plot(2, 1, '*b')


% dim = 2;
% chi = 1;
% beta = log(5);
% plotting = true;
% 
% nbr_observations = 8;
% 
% capability_mean = zeros(dim, 1);
% capability_cov = eye(dim);
% 
% 
% observations = 10 * rand(dim, nbr_observations) - 5.0;
% 
% S = 1;
% 
% for i = 1:length(observations)
%     
%     capability_mean(:, i + 1) = (1/S) * (observations(:, i) + exp(-beta) * capability_mean(:, i));
%     S = S + exp(-beta) * S;
%    
% 
%     if plotting
%         plot(capability_mean(1,i), capability_mean(2,i), 'b*')
%         hold on
%         axis([-5 5 -5 5])
%         grid on
%         pause(0.5)
%         plot(observations(1,1:i), observations(2,1:i), 'g*')
%         pause(0.5)
%     end    
%     
%     
% end





% obs1 = [1.5; 1.0];
% 
% S = 1 + exp(-beta);
% 
% capability_mean = obs1/S;
% 
% plot(capability_mean(1), capability_mean(2), '*')



% if dim == 2
%     plot_ellip(capability_mean, capability_cov, chi)
% end





% mu = [1; 1];
% t = linspace(-.99, 0.99, 10);
% 
% for i = 1:length(t)
%     sigma = [1 t(i); t(i) 1];
%     chi = 1;
% 
%     clf
%     plot_ellip(mu, sigma, chi)
%     pause(0.1)
% end