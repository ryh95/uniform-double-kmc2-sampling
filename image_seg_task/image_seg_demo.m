
image_dir = '../data/jpg_images/';

% read image, change color image to brightness image, resize to 160x160
I = imread_ncut(strcat(image_dir,'3.jpg'),80,80);

% % display the image
% figure(1);clf; imagesc(I);colormap(gray);axis off;
% disp('This is the input image to segment, press Enter to continue...');

% compute the edges imageEdges, the similarity matrix W based on
% Intervening Contours, the Ncut eigenvectors and discrete segmentation
nbSegments = 5;
disp('computing Ncut eigenvectors ...');
tic;
[SegLabel,NcutDiscrete,NcutEigenvectors,NcutEigenvalues,A,imageEdges]= NcutImage(I,nbSegments);
disp(['The computation took ' num2str(toc) ' seconds on the ' num2str(size(I,1)) 'x' num2str(size(I,2)) ' image']);

% make A matrix sparse
valeurMin = 1e-6;
A = sparsifyc(A,valeurMin);

% check for matrix symmetry
if max(max(abs(A-A'))) > 1e-10 %voir (-12) 
    %disp(max(max(abs(A-A'))));
    error('A is not symmetric');
end

n = size(A,1);

% degrees and regularization
offset = 5e-1;
d = sum(abs(A),2);
dr = 0.5 * (d - sum(A,2));
d = d + offset * 2;
dr = dr + offset;
A = A + spdiags(dr,0,n,n);

D = sum(A,2);

Z = NcutDiscrete*diag(diag(1./sqrt(NcutDiscrete'*diag(D)*NcutDiscrete)));
Nassoc = trace(Z'*A*Z);

% % display the edges
% figure(2);clf; imagesc(imageEdges); axis off
% disp('This is the edges computed, press Enter to continue...');

% % display the segmentation
Ncut = nbSegments - sum(NcutEigenvalues);
figure(3);clf
bw = edge(SegLabel,0.01);
J1=showmask(I,imdilate(bw,ones(2,2))); imagesc(J1);axis off
disp('This is the segmentation, press Enter to continue...');