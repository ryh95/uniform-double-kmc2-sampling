% join all rna data
fileID = fopen(strcat('../data/','cod-rna','.txt'),'r');
[A_1,~] = fscanf(fileID, ['%d' repmat('%*d:%f',1,8)] ,[9 Inf]);
fclose(fileID);
fileID = fopen(strcat('../data/','cod-rna.t','.txt'),'r');
[A_2,~] = fscanf(fileID, ['%d' repmat('%*d:%f',1,8)] ,[9 Inf]);
fclose(fileID);
fileID = fopen(strcat('../data/','cod-rna.r','.txt'),'r');
[A_3,~] = fscanf(fileID, ['%d' repmat('%*d:%f',1,8)] ,[9 Inf]);
fclose(fileID);

A = [A_1 A_2 A_3];
% change precision if you need
dlmwrite('rna.txt',A');