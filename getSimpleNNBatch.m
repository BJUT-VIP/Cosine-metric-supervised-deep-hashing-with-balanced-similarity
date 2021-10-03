function [images, labels] = getSimpleNNBatch(imdb, batch)
images = imdb.data(:, :, :, batch) ;
% if rand > 0.5, images=fliplr(images) ; end
if isempty(imdb.labels)
    labels = [];
else
    labels = imdb.labels(:, batch);
end
end