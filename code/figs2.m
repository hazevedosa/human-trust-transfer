close all; clear all; clc;


x = 0:0.001:1;
y = 0:0.001:1;

beta = [100];
zeta = [100];


figure(1)
for i = 1:length(beta)
    
    t = 1 ./ (1 + exp(beta(i) * (x-0.5)));

    plot(x, t)
    hold on
    
end

% legend('beta = 0.25', 'beta = 1', 'beta = 10', 'beta = 100', 'beta = 1000')
axis([min(x) max(x) 0 1])


figure(2)
for i = 1:length(zeta)
    
    t2 = (1 ./ (1 + exp(10 * (x-0.5)))).^zeta(i);

    plot(x, t2)
    hold on
    
end

% legend('beta = 0.25', 'beta = 1', 'beta = 10', 'beta = 100', 'beta = 1000')
axis([min(x) max(x) 0 1])

figure(3)

for i = 1:length(zeta)
    for j = 1:length(beta)

        t3 = (1 ./ (1 + exp(beta(j) * (x-0.5)))).^zeta(i);

        plot(x, t3)
        hold on
    end
end

% legend('beta = 0.25', 'beta = 1', 'beta = 10', 'beta = 100', 'beta = 1000')
axis([min(x) max(x) 0 1])
