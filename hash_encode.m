function [U, H, HLabels]= hash_encode(net, batchFunc, imdb, codelens)
verbose = true;
myLogInfo('Testing nus on %d images ...', size(imdb.labels, 2)); 
t0 = tic; tic;

U = zeros(codelens, size(imdb.labels, 2));
H = zeros(codelens, size(imdb.labels, 2), 'single');  %制作存储哈希矩阵，bit*test/train的全零矩阵，类型为single,32*2100
HLabels = zeros(size(imdb.labels, 1), size(imdb.labels, 2));
batch_size = 128;
for t = 1:batch_size:size(imdb.labels, 2)  %按batch_size来读取。
    ed = min(t+batch_size-1, size(imdb.labels, 2)); 
    [data, labels] = batchFunc(imdb, t:ed); %按batch_size长度来读取文件列表中的数据
    data = gpuArray(data);   %把数据转换为gpu数据，即把数据从MATLAB工作空间传送到GPU上
    res = vl_simplenn(net, data);
    rex = squeeze(gather(res(end).x));%res(end).x 表示最后一层，gather为把GPU上的数据转换成CPU数据，squeeze为删除单一维度

    U(:, t:ed) = rex;
    H(:, t:ed) = single(sign(rex) > 0);   %把数据变为单精度数据。
    HLabels(:, t:ed) = labels;
    if verbose && toc > 100
        myLogInfo('%6d / %d', t, size(imdb.labels, 2)); 
        tic;
    end
end
if verbose, toc(t0); end
end
