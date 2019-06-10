function show_seg(I,P)
    [nr,nc,nb] = size(I);

    SegLabel = zeros(nr,nc);
    for j=1:size(P,2)
        SegLabel = SegLabel + j*reshape(P(:,j),nr,nc);
    end
    % display the segmentation
    figure(3);clf
    bw = edge(SegLabel,0.01);
    J1=showmask(I,imdilate(bw,ones(2,2))); imagesc(J1);axis off