function imdb = get_imdb(imdb, net, name)
imdbName = name;  %������imdbName
imdbFunc = str2func(imdbName);   %imdbFunc����imdb�µ�nus.m������
imdb = [];

% imdbFile
imdbName = sprintf('%s_split%d', imdbName);
imdbFile = fullfile('./data/', ['imdb_' imdbName '.mat']); %����imdbFile��·����.mat�ļ�
    if strcmp(name, 'nus')
        dataDir = 'E:/CMDH_NDCG/data';
    else
        dataDir = 'E:/CMDH_NDCG/data';
    end
% load/save
t0 = tic;
if exist(imdbFile, 'file')
    imdb = load(imdbFile) ;
    myLogInfo('loaded in %.2fs', toc(t0));
else
    imdb = imdbFunc(dataDir, net) ;
    save(imdbFile, '-struct', 'imdb', '-v7.3') ; %����ṹ��?
    myLogInfo('saved in %.2fs', toc(t0)); %����洢ʱ��?
end
imdb.name = imdbName;
myLogInfo('%s loaded', imdb.name); %��������imdb����
end
