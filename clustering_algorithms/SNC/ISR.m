function [Y, obj,changed] = ISR(A, c,NITR)
% A: similarity matrix
% c: number of clusters
% k: number of neighbors to determine the initial graph
% NITR: max iteratios to run

% Y: the final clustering result, cluster indicator vector


if nargin<3
    NITR=30;
end

DD=diag(sum(A));
if issparse(DD)
    D2=sparse(full(DD)^(-0.5));
else
    D2=DD^(-0.5);
end
[F, ~, ~]=eig1(D2*A*D2, c, 1);

% Initialize Y
Y=diag(diag(F*F'))^(-0.5)*F;
for i=1:size(Y,1)
    [~,mix]=max(Y(i,:));
    Y(i,:)=0;
    Y(i,mix)=1;
end

[Y,obj,changed]=SR_exact(F, DD, Y,NITR);

end

