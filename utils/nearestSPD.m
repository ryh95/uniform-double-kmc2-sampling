function Ahat = nearestSPD(A)
% nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
% usage: Ahat = nearestSPD(A)
%
% From Higham: "The nearest symmetric positive semidefinite matrix in the
% Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
% where H is the symmetric polar factor of B=(A + A')/2."
%
% http://www.sciencedirect.com/science/article/pii/0024379588902236
%
% arguments: (input)
%  A - square matrix, which will be converted to the nearest Symmetric
%    Positive Definite Matrix, can be a sparse matrix
%
% Arguments: (output)
%  Ahat - The matrix chosen as the nearest SPD matrix to A.

if nargin ~= 1
  error('Exactly one argument must be provided.')
end

% test for a square matrix A
[r,c] = size(A);
if r ~= c
  error('A must be a square matrix.')
elseif (r == 1) && (A <= 0)
  % A was scalar and non-positive, so just return eps
  Ahat = eps;
  return
end

% symmetrize A into B
B = (A + A')/2;

% Compute the symmetric polar factor of B. Call it H.
% Clearly H is itself SPD.

% TODO: speed up svd, may be use svds?
[~,Sigma,V] = svd(full(B));
% [U,Sigma,V] = svds(B,20);
Sigma = sparse(Sigma);

% TODO: speed up the following matrix multiplication, also try to reduce
% the memory
H = V*Sigma*V';
H = sparse(H);
clear V;

% [V,D] = eig(B'*B);
% H = V*sqrt(D)*V';

% [U,Sigma,V_svd] = svd(B);
% H_svd = V_svd*Sigma*V_svd';

% get Ahat in the above formula
Ahat = (B+H)/2;
clear H;
% ensure symmetry
Ahat = (Ahat + Ahat')/2;

% test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
p = 1;
k = 0;
while p ~= 0
  [R,p] = chol(Ahat);
  k = k + 1;
  if p ~= 0
    % Ahat failed the chol test. It must have been just a hair off,
    % due to floating point trash, so it is simplest now just to
    % tweak by adding a tiny multiple of an identity matrix.
    
    % TODO: find min eigen value more efficiently
    % I tried the following, but got an error
    % Warning: Matrix is close to singular or badly scaled, solve this in
    % the future
    %     mineig_eigs = eigs(Ahat,1,'SM');
    % another issiue is eig on sparse matrix seems to be very slow, figure
    % it out in the future
    mineig = min(eig(full(Ahat)));
    Ahat = Ahat + (-mineig*k.^2 + eps(mineig))*speye(size(A));
  end
end






