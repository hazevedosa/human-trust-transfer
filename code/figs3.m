close all; clear all; clc;

n_bins = 100;

l1 = 0.5;
u1 = 1.0;

l2 = 0.5;
u2 = 1.0;

area = (u1 - l1) * (u2 - l2);
alpha = 1 - (0.5 + 0.5 / (n_bins * n_bins * area))

poly_Xs = [l1 u1 u1 l1];
poly_Ys = [l2 l2 u2 u2];

pgon = polyshape(poly_Xs, poly_Ys)

shadows = plot(pgon);
shadows.FaceColor = [0.25 0.25 0.25];
shadows.FaceAlpha = 1;
shadows.LineStyle = 'none';

axis equal
axis([0 1 0 1])