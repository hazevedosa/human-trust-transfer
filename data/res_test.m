close all; clear all; clc;

MAE1 = [0.15885712,
0.24112833,
0.3576719,
0.34587356,
0.25498697,
0.21495606,
0.25932822,
0.39580014,
0.25168633,
0.27160165];

PL1 = [
    0.6819172501564026,
0.5830984115600586,
0.8044256567955017,
0.7127425074577332,
0.6625885963439941,
0.5824704170227051,
0.6511467099189758,
0.7953051924705505,
0.5344669222831726,
0.7383669018745422];

MAE2 = [
    0.29475582,
0.2931061,
0.41120315,
0.3646091,
0.31126186,
0.16250747,
0.26797104,
0.34613523,
0.25042847,
0.20463581];

PL2 = [
    0.6494774222373962,
0.6260694861412048,
0.9455899596214294,
0.817406415939331,
0.7105321884155273,
0.5385445356369019,
0.6486513018608093,
0.7246285080909729,
0.5504072308540344,
0.6243446469306946];


MAE3 = [0.2623984,
0.25863802,
0.07382821,
0.2686844,
0.42701283,
0.15560205,
0.0857901,
0.18828624,
0.19331263,
0.12562048];

PL3 = [0.4980619251728058,
0.7480816841125488,
0.4405719041824341,
0.7501489520072937,
0.9385032653808594,
0.5286812782287598,
0.41279423236846924,
0.49257341027259827,
0.6029792428016663,
0.45603927969932556];


mean(MAE1)
mean(MAE2)
mean(MAE3)

std(MAE1)
std(MAE2)
std(MAE3)
