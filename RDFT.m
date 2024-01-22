function [A, Ps, Pt, R, yt_pred, acc, w] = RDFT(Xs_train, ys_train, Xt_train, yt_train, param, opts)
lambda1 = param.lambda1;
lambda2 = param.lambda2; 
lambda3 = param.lambda3;
lambda4 = param.lambda4;
lambda5 = param.lambda5;

dim = opts.dim;
gamma = opts.gamma;
rho = opts.rho;
tau = opts.tau;
delta = opts.delta;
rho_max = opts.rho_max;
epsilon = opts.epsilon;
max_iter = opts.max_iter;

[d, ns] = size(Xs_train);
nt=size(Xt_train,2);
C=length(unique(ys_train));
I=eye(C);
Ys=I(:,ys_train);

Ps=zeros(d,dim);
Pt=zeros(d,dim);
R=zeros(C,dim);
Z=zeros(ns,nt);
B=zeros(ns,nt);
 
N=ones(C,ns);
S = 2 * Ys - ones(C, ns);

w=gamma*ones(ns,1);
U1=zeros(C,d);
U2=zeros(ns,nt);
y_lbl=unique(ys_train);

for i = 1:C
    ns_c = sum(ys_train == y_lbl(i));
    g(i,1)=gamma*ns_c;
end

mdl= fitcknn(Xs_train', ys_train, 'NumNeighbors',1);
yt_pred = predict(mdl, Xt_train');
accuracy=sum(yt_train == yt_pred)/length(yt_pred)*100;
fprintf('acc of non-transfer learning: %.2f \n', accuracy);
acc=[];

for iter=1:max_iter
    RPt_prev=R*Pt';
    
    [m,MNss,MNtt,MNst] = matrixMN(ns,nt,ys_train,yt_pred,gamma);
    Hss = (w*w').*MNss;
    Hst = -(w*ones(nt,1)').*MNst; 
    Hts = Hst';
    Htt = (ones(nt,1)*ones(nt,1)').*MNtt;
    
    %update Q
    [U,~,V]=svd(Xt_train*(Pt'*Xt_train)','econ');
    Q=U*V';
    
    %update A
    T1=Xs_train*diag(w)*Xs_train'+rho*eye(d);
    T2=(Ys+S.*N)*diag(w)*Xs_train'+rho*(R*Ps'+U1/rho);
    A=T2/T1;

    %update Ps
    T3=lambda2*Xs_train*Z*Z'*Xs_train'+lambda4*Xs_train*Hss*Xs_train'+lambda5*eye(d);
    T4=rho*R'*R;
    T5=lambda2*Xs_train*Z*Xt_train'*Pt-lambda4*Xs_train*Hst*Xt_train'*Pt+lambda5*Pt+rho*(A-U1/rho)'*R;
    Ps=sylvester(T3,T4,T5);
    
    %update Pt
    T6=(lambda1+lambda2)*Xt_train*Xt_train'+lambda4*Xt_train*Htt*Xt_train'+lambda5*eye(d);
    T7=lambda1*Xt_train*Xt_train'*Q+lambda2*Xt_train*Z'*Xs_train'*Ps-lambda4*Xt_train*Hts*Xs_train'*Ps+lambda5*Ps;
    Pt=T6\T7;
    
    %update R
    R=rho*(A-U1/rho)*Ps/(rho*Ps'*Ps+tau*eye(dim)); %inv

    %update w
    F=[Xs_train'*Ps; Xt_train'*Pt] * [Ps'*Xs_train, Pt'*Xt_train];
    F1=F.*m;
    E=Ys+S.*N-A*Xs_train;
    Vss=F1(1:ns,1:ns);
    Vts=F1(ns+1:end, 1:ns);
    T8=diag(E'*E)-Vts'*ones(nt,1);
    lb = zeros(ns,1);
    ub = ones(ns,1);
    options = optimoptions('quadprog','Display','off');
    warning('off','all');
    w= quadprog(double(Vss),double(T8), [], [], Ys, g, lb, ub, [], options);
    warning('on','all');
    
    %update N
    N=max(0,(A*Xs_train-Ys).*S);
    
    %update Z
    T9=lambda2*Xs_train'*Ps*Ps'*Xs_train+rho*eye(ns);
    T10=lambda2*Xs_train'*Ps*Pt'*Xt_train+rho*(B-U2/rho);
    Z=T9\T10;
    
    %update B
    B=sign(Z+U2/rho).*max(0,abs(Z+U2/rho)-lambda3/rho);
    
    %update Us and rho
    U1=U1+rho*(R*Ps'-A);
    U2=U2+rho*(Z-B);
    rho=min(delta*rho, rho_max);
    fea_test=R*Pt'*Xt_train;
    [~,pred]=max(fea_test);
    yt_pred=pred';
    
    accuracy=sum(yt_train == yt_pred)/length(yt_pred)*100;
    acc=[acc accuracy];
    fprintf('%d / %d  acc: %.2f \n', iter, max_iter, accuracy);
    
    fprintf('RPt_delta2=%.5f \n', norm(R*Pt'-RPt_prev,'fro')/norm(RPt_prev,'fro'))
    if norm(R*Pt'-RPt_prev,'fro')/norm(RPt_prev,'fro') < epsilon
        break;
    end
end 