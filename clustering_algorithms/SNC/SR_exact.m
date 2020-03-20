% min_{G=Ind,Q'*Q=I} || D^{0.5}*G*(G'*D*G)^{-0.5} - F0*Q ||^2
function [Gsr4, obj_sr4,changed] = SR_exact(F0, D, G,NITR)

[n,class_num] = size(F0);
D2 = diag(diag(D).^0.5);

if nargin < 3
    StartInd = randsrc(n,1,1:class_num);
    G = n2nc(StartInd);
end;

if nargin<4
    NITR=30;
end
Gsr4 = G;
obj_sr4=zeros(NITR,1);
changed=zeros(NITR,10);
for iter = 1:NITR
    Gd = D2*Gsr4*(Gsr4'*D*Gsr4+eps*eye(class_num))^-0.5;
    [U, ~, V] = svd(Gd'*F0);
    Q = V*U';
    clear U V;
    FQ = F0*Q;
    [~, g] = max(FQ,[],2);
    Gsr4 = TransformL(g, class_num);
    nn = sum(Gsr4.*(D*Gsr4));
    s = sum(FQ.*(D2*Gsr4));
    [fq] = max(FQ,[],2);
    [~,idxi] = sort(fq);
    for it = 1:10
        converged=true;
        for ii = 1:n
            i = idxi(ii);
            dd = D(i,i);
            dd2 = D2(i,i);
            hi = FQ(i,:);
            gi = Gsr4(i,:);
            [~,id0] = max(gi);
            ss = (s+(1-gi).*(dd2*hi))./sqrt(nn+(1-gi)*dd) - (s-gi.*(dd2*hi))./(sqrt(nn-gi*dd)+eps);
            [~,id] = max(ss);
            if id~=id0
                converged=false;
                changed(iter,it)=changed(iter,it)+1;
                gi = zeros(1,class_num);
                gi(id) = 1;
                Gsr4(i,:) = gi;
                nn(id0) = nn(id0) - dd;  nn(id) = nn(id) + dd;
                s(id0) = s(id0) - dd2*FQ(i,id0);  s(id) = s(id) + dd2*FQ(i,id);
            end;
        end;
        if converged
            clear s nn;
            break;
        end
    end;
    Gd = D2*Gsr4*(Gsr4'*D*Gsr4+eps*eye(class_num))^-0.5;
    obj_sr4(iter) = trace((Gd-FQ)'*(Gd-FQ));
    if iter>2 && abs((obj_sr4(iter)-obj_sr4(iter-1))/obj_sr4(iter))<1e-10
        break;
    end
end;



