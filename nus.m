function DB = nus(dataDir, net)

sdir = fullfile(dataDir, 'NUS-WIDE');
% sdir = 'F:/dataset/Flickr'

images = textread([sdir '/Imagelist.txt'], '%s');
images = strrep(images, '\', '/');
images = strrep(images, 'C:/ImageData', 'F:/dataset');

% get labels
labels = load([sdir '/AllLabels81.txt']);
myLogInfo('Total images = %g', size(labels, 1));

% use 21 most frequent labels only
myLogInfo('Keeping 21 most frequent tags, removing rest...');
[val, sel] = sort(sum(labels, 1), 'descend');
labels = labels(:, sel(1:21));
myLogInfo('Min tag freq %d', val(21));

% remove those without any labels
keep   = sum(labels, 2) > 0;
labels = labels(keep, :);
images = images(keep);
assert(size(labels, 1) == length(images));
myLogInfo('Keeping # images = %g', sum(keep));

% split
sets = split_nus(labels);

% save
train_num = find(sets == 1);
test_num = find(sets == 3);
database_num = find(sets == 1 | sets == 2);

meanImage = single(net.meta.normalization.averageImage);
imgSize = net.meta.normalization.imageSize(1);
sz = [imgSize imgSize];
if isequal(size(meanImage), [1 1 3])
    meanImage = repmat(meanImage, sz);
else
    assert(isequal(size(meanImage), [sz 3]));
end
imgs = vl_imreadjpeg(images(train_num), 'Pack', 'NumThreads', 4, ...
                    'Resize', sz, 'subtractAverage', meanImage);
imgs = imgs{1};
size(imgs)
DB.traintxt.data = images(train_num);
DB.traintxt.labels = single(labels(train_num, :))';
DB.train.data = imgs;
DB.train.labels =  single(labels(train_num, :))';
DB.test.data = images(test_num);
DB.test.labels = single(labels(test_num, :))';
DB.database.data = images(database_num);
DB.database.labels = single(labels(database_num, :))';

% DB.train.data = images(train_num);
% DB.train.labels =  single(labels(train_num, :))';
% DB.test.data = images(test_num);
% DB.test.labels = single(labels(test_num, :))';
% DB.database.data = images(database_num);
% DB.database.labels = single(labels(database_num, :))';

% DB.images.data = images;  % only save image names, load on demand
% DB.images.labels = single(labels)';
% DB.images.set = uint8(sets)';
% DB.meta.sets = {'train', 'val', 'test'} ;
end
