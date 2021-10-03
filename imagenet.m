function DB = imagenet(dataDir, net)

sdir = fullfile(dataDir, 'ImageNet');
sdir1 = 'F:';
% train
[img_train, cls_train] = readHashNetTxt(fullfile(sdir, 'train.txt'), sdir1);
myLogInfo('Train: %d imgs', numel(img_train));

% test/database
% img_test = [];
% cls_test = [];
% img_all  = [];
% cls_all  = [];
% for i = 0:7
%     [It, Ct] = readHashNetTxt(fullfile(sdir, sprintf('test%d.txt', i)), sdir1);
%     img_test = [img_test; It];
%     cls_test = [cls_test; Ct];
% 
%     [Ib, Cb] = readHashNetTxt(fullfile(sdir, sprintf('database%d.txt', i)), sdir1);
%     img_all  = [img_all; Ib];
%     cls_all  = [cls_all; Cb];
% end
[img_test, cls_test] = readHashNetTxt(fullfile(sdir,'test.txt'), sdir1);
[img_all, cls_all] = readHashNetTxt(fullfile(sdir, 'database.txt'), sdir1);
   
myLogInfo(' Test: %d imgs', numel(img_test));
myLogInfo('  ALL: %d imgs', numel(img_all));

% read train images
meanImage = single(net.meta.normalization.averageImage);
imgSize = net.meta.normalization.imageSize(1);
sz = [imgSize imgSize];
if isequal(size(meanImage), [1 1 3])
    meanImage = repmat(meanImage, sz);
else
    assert(isequal(size(meanImage), [sz 3]));
end
% imgs = vl_imreadjpeg(img_train, 'Verbose', 'Pack', 'NumThreads', 8, ...
%     'Interpolation', 'bicubic', 'Resize', sz, 'subtractAverage', meanImage);
imgs = vl_imreadjpeg(img_train, 'Pack', 'NumThreads', 4, ...
                    'Resize', sz, 'subtractAverage', meanImage);
imgs = imgs{1};
%imgs = cell(1, numel(img_train));
%tic;
%for i = 1:numel(img_train)
%    im = imresize(imread(img_train{i}), sz);
%    if toc>10
%        myLogInfo('reading train imgs: %d/%d', i, numel(img_train));
%        tic;
%    end
%end
%%imgs = cellfun(@(x) x-meanImage, imgs, 'Uni', false);
%imgs = cat(4, imgs{:});
size(imgs)

% set
% assert(all(ismember(img_train, img_all)));
% set_all = 2*ones(numel(img_all), 1);
% set_all(ismember(img_all, img_train)) = 1;
% set_all = [set_all; 3*ones(numel(img_test), 1)];

% save
DB.traintxt.data = img_train;
DB.traintxt.labels =  single(cls_train)';
DB.train.data = imgs;
DB.train.labels =  single(cls_train)';
DB.test.data = img_test;
DB.test.labels = single(cls_test)';
DB.database.data = img_all;
DB.database.labels = single(cls_all)';
% imdb.images.data = imgs;
% imdb.images.labels = single(cls_train)';
% imdb.images.set = ones(1, numel(cls_train), 'uint8');
% imdb.images.all = [];
% imdb.images.all.data = [img_all; img_test];  % load on demand
% imdb.images.all.labels = single([cls_all; cls_test])';
% imdb.images.all.set = uint8(set_all);
% imdb.meta.sets = {'train', 'val', 'test'} ;
end


function [img, cls] = readHashNetTxt(fn, sdir)
%fid = fopen(fullfile(sdir, 'train.txt'));
fid = fopen(fn);
[c] = textscan(fid, ['%s' repmat(' %d', 1, 100)]);
fclose(fid);
img = strrep(c{1}, '/home/caozhangjie/run-czj', sdir);
cls = cat(2, c{2:end});
% [r, c] = find(cls);
% cls = [];  
% cls(r) = c;
% cls = cls';
end
