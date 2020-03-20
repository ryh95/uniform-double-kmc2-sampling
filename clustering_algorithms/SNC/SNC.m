function [Y, obj,changed] = SNC(X, c,m,k,NITR)
% X: [d n] data matrix, where n is the number of objects and d is the
% c: number of clusters
% m: number of anchors
% k: number of neighborhoods
% NITR: max iteratios to run

% Y: the final clustering result, cluster indicator vector

if nargin<5
    NITR=30;
end
n=size(X,2);
B = constructAnchorDistance_PKN(X,m, k);
P=B*diag(sum(B,1))^-0.5;
[U,Lambda,~] = svd(P);
[~,ind]=sort(diag(Lambda).^2,'descend');
F=U(:,ind(1:c));

% Initialize Y
Y=diag(diag(F*F'))^(-0.5)*F;
for i=1:size(Y,1)
    [~,mix]=max(Y(i,:));
    Y(i,:)=0;
    Y(i,mix)=1;
end

Gsr4 = Y;
obj=zeros(NITR,1);
changed=zeros(NITR,10);
for iter = 1:NITR
    Gd = Gsr4*(Gsr4'*Gsr4+eps*eye(c))^-0.5;
    [U, ~, V] = svd(Gd'*F);
    Q = V*U';
    clear U V;
    FQ = F*Q;
    [~, g] = max(FQ,[],2);
    Gsr4 = TransformL(g, c);
    nn = sum(Gsr4.*(Gsr4));
    s = sum(FQ.*(Gsr4));
    [fq] = max(FQ,[],2);
    [~,idxi] = sort(fq);
    for it = 1:10
        converged=true;
        for ii = 1:n
            i = idxi(ii);
            hi = FQ(i,:);
            gi = Gsr4(i,:);
            [~,id0] = max(gi);
            ss = (s+(1-gi).*hi)./sqrt(nn+(1-gi)) - (s-gi.*hi)./(sqrt(nn-gi)+eps);
            [~,id] = max(ss);
            if id~=id0
                converged=false;
                changed(iter,it)=changed(iter,it)+1;
                gi = zeros(1,c);
                gi(id) = 1;
                Gsr4(i,:) = gi;
                nn(id0) = nn(id0) - 1;  nn(id) = nn(id) + 1;
                s(id0) = s(id0) - FQ(i,id0);  s(id) = s(id) + FQ(i,id);
            end;
        end;
        if converged
            clear s nn;
            break;
        end
    end;
    Gd = Gsr4*(Gsr4'*Gsr4+eps*eye(c))^-0.5;
    obj(iter) = trace((Gd-FQ)'*(Gd-FQ));
    if iter>2 && abs((obj(iter)-obj(iter-1))/obj(iter))<1e-10
        break;
    end
end;

