close all; clear all; clc;

b = 500;

l1 = 0.3;
u1 = 0.5;
l2 = 0.5;
u2 = 0.8;

dl = 0.025;

vv = 0.0:dl:1.0;

[L1, L2] = meshgrid(vv, vv);

T = trust(l1, u1, b, L1) .* trust(l2, u2, b, L2);
surf(L1, L2, T, 'FaceAlpha',0.5)