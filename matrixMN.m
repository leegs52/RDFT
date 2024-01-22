function [m,MNss,MNtt,MNst] = matrixMN(ns,nt,ys_train,yt_pred,gamma)

lbl_idx = unique(ys_train);
C = length(lbl_idx);

tm = ones(ns+nt,1);
tm(1:ns,1) = 1/(gamma*ns);
tm(ns+1:end,1) = 1/nt;
theta_m = tm*tm';

theta_c1 = zeros(ns+nt,ns+nt);
for i = 1:C
    Sc_index = ys_train == lbl_idx(i);
    Tc_index = yt_pred == lbl_idx(i);
    nsc = sum(Sc_index);
    ntc = sum(Tc_index);
    if ntc == 0 || nsc ==0
        continue;
    end
    
    theta = zeros(ns+nt,1);
    theta(Sc_index) = (1/(gamma*nsc))*ones(nsc,1);
    theta(ns+find(Tc_index)) = (1/ntc)*ones(ntc,1);
    theta_c1 = theta_c1 + theta*theta';
end
m = theta_m + theta_c1;
MNss = m(1:ns,1:ns);
MNtt = m(ns+1:end,ns+1:end);
MNst = m(1:ns, ns+1:end);
end