function [images, labels] = trainbatch_imagenet(imdb, batch, imgSize, meanImage)
if ~iscell(imdb.data)
    % already loaded in imdb 判断是否是元胞数组，如果不是，进行下面这段语句
    images = imdb.data(:, :, :, batch) ;
    % normalization
    if imgSize ~= size(images, 1)
        images = imresize(images, [imgSize, imgSize]); %缩放到224*224大小
    end
    meanImage = repmat(meanImage,1,1,1,numel(batch));
    images = bsxfun(@minus, images, meanImage); %图像减去均值
    % get labels
    if isempty(imdb.labels)
        itrain = find(imdb.images.set == 1);
        [~, labels] = ismember(batch, itrain);
    else
        labels = imdb.labels(batch);
    end   
else
%     meanImage = repmat(meanImage,1,1,1,numel(batch));
%     args = {'Gpu', 'Pack', ...
%                 'NumThreads', 4, ...
%                 'Resize', [imgSize imgSize], ...
%                 'subtractAverage', meanImage};
    % train or test? train: use data augmentation
%     if imdb.images.set(batch(1)) == 1
%         args{end+1} = 'Flip';
%     end
    % imdb.images.data is a cell array of filepaths
    % first call: prefetch
    vl_imreadjpeg(imdb.data(batch), 'Verbose', 'Pack', 'NumThreads', 8, ...
    'Interpolation', 'bicubic', 'Resize', [imgSize, imgSize], 'subtractAverage', meanImage);
    % get labels now
        if isempty(imdb.labels)
            assert(~isempty(imdb.labels),  disp('没有训练集标签'));
        else
%         labels = permute(imdb.images.labels(:, batch), [3 4 2 1]);
            labels = imdb.labels(:, batch);
        end
    % second call to actually get images
    images = vl_imreadjpeg(imdb.data(batch), 'Verbose', 'Pack', 'NumThreads', 8, ...
    'Interpolation', 'bicubic', 'Resize', [imgSize, imgSize], 'subtractAverage', meanImage);
    images = images{1}; 
    if rand > 0.5, images=fliplr(images) ; end
end
end
% get images
% if ~iscell(imdb.images.data)
%     % already loaded in imdb 判断是否是元胞数组，如果不是，进行下面这段语句
%     images = imdb.images.data(:, :, :, batch) ;
%     % normalization
%     if imgSize ~= size(images, 1)
%         images = imresize(images, [imgSize, imgSize]); %缩放到224*224大小
%     end
%     images = bsxfun(@minus, images, meanImage); %图像减去均值
%     % get labels
%     if isempty(imdb.images.labels)
%         itrain = find(imdb.images.set == 1);
%         [~, labels] = ismember(batch, itrain);
%     else
%         labels = permute(imdb.images.labels(:, batch), [3 4 2 1]);
%     end
% else
%     args = {'Gpu', 'Pack', ...
%             'NumThreads', 4, ...
%             'Resize', [imgSize imgSize], ...
%             'Interpolation', 'bicubic', ...
%             'subtractAverage', meanImage};
%     % train or test? train: use data augmentation
%     if imdb.images.set(batch(1)) == 1
%         args{end+1} = 'Flip';
%     end
%     % imdb.images.data is a cell array of filepaths
%     % first call: prefetch
%     vl_imreadjpeg(imdb.images.data(batch), args{:}, 'prefetch');
%     % get labels now
%     if isempty(imdb.images.labels)
%         itrain = find(imdb.images.set == 1);
%         [~, labels] = ismember(batch, itrain);
%     else
% %         labels = permute(imdb.images.labels(:, batch), [3 4 2 1]);
%         labels = imdb.images.labels(:, batch);
%     end
%     % second call to actually get images
%     images = vl_imreadjpeg(imdb.images.data(batch), args{:});
%     images = images{1};
% end


