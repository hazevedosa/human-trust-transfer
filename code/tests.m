% just thinking...

close all; clear all; clc;


n = 3;

for i = 1:10^n
    u = mod(i-1, 10)
    d = mod(i-1, 100)
    m = mod(i-1, 1000)
end