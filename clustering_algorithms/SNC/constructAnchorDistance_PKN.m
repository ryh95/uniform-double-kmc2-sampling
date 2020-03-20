function z = constructAnchorDistance_PKN( X, m,k, nRepeat)
%KNNDISTANCE Summary of this function goes here
%   Detailed explanation goes here

% X: each column is a data point
% anchors: each column is a data point
% nRepeat: the execute time to repeat



if nargin<4
    nRepeat=10;
end

[~,n]=size(X);
obj_Old=realmax;
for jj = 1 : nRepeat
    % in large folder, change filename because name conflict
    [label, ~, ~, center, obj] = kmeans1(X, randsrc(n, 1, 1:m));
    if obj<obj_Old
        ct=center;
        lb=label;
        obj_Old=obj;
    end
end;

ub=unique(lb);

anchors=zeros(size(X,1),m);
len=length(ub);
for i=1:len
    anchors(:,i)=ct(:,ub(i));
end

left=m-len;
if left>0
    ns=tabulate(lb);
    ns=ns(:,3)./100;
    ns=floor((m-len)*ns);
    left=left-sum(ns);
    while left>0
        if left>len
            ns=ns+1;
            left=left-len;
        else
            sl=randsample(len,left,0);
            for i=1:size(sl,1)
                l=sl(i);
                ns(l)=ns(l)+1;
            end
            break;
        end
    end
    
    ptr=len+1;
    for i=1:len
        if ns(i)>0
            sl=randsample(find(lb==i),ns(i));
            for j=1:length(sl)
                anchors(:,ptr)=X(:,sl(j));
                ptr=ptr+1;
            end
        end
    end
end

D = L2_distance(X,anchors);
D(D<0) = 0;

if nargin<3 || k>=size(anchors,2)
    k=size(anchors,2)-1;
end

[~ , idx] = sort(D, 2); % sort each row
num=size(X,2);
z = zeros(num,m);
for i = 1:num
    id = idx(i,1:k);
    di = D(i, idx(i,1:k+1));
    z(i,id) = (di(k+1)-di(1:k))/(k*di(k+1)-sum(di(1:k)));
end;

z(isnan(NaN))=1/m;
clear D idx;

end

