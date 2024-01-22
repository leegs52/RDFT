clear all
Xys= csvread('./data/office31_resnet50/amazon_amazon.csv');
Xs=Xys(:,1:2048)';
ys=Xys(:,2049)+1;
Xyt= csvread('./data/office31_resnet50/amazon_dslr.csv');
Xt=Xyt(:,1:2048)';
yt=Xyt(:,2049)+1;

param.lambda1 = 0.0001;
param.lambda2 = 0.1;
param.lambda3 = 0.01;
param.lambda4 = 50;
param.lambda5 = 1;

opts.dim = round(size(Xs,1)/4);
opts.gamma=0.99;
opts.rho = 0.1;
opts.tau = 0.001;
opts.delta = 1.01;
opts.rho_max = 1e5;
opts.epsilon = 1e-3;
opts.max_iter= 30;

Xs_train = Xs ./repmat(sqrt(sum(Xs .*Xs)), [size(Xs , 1), 1]);
Xt_train = Xt ./repmat(sqrt(sum(Xt .*Xt)), [size(Xt , 1), 1]);
ys_train=ys;
yt_train=yt;

[A, Ps, Pt, R, yt_pred, acc, w] = RDFT(Xs_train, ys_train, Xt_train, yt_train, param, opts);