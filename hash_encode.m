function [U, H, HLabels]= hash_encode(net, batchFunc, imdb, codelens)
verbose = true;
myLogInfo('Testing nus on %d images ...', size(imdb.labels, 2)); 
t0 = tic; tic;

U = zeros(codelens, size(imdb.labels, 2));
H = zeros(codelens, size(imdb.labels, 2), 'single');  %�����洢��ϣ����bit*test/train��ȫ���������Ϊsingle,32*2100
HLabels = zeros(size(imdb.labels, 1), size(imdb.labels, 2));
batch_size = 128;
for t = 1:batch_size:size(imdb.labels, 2)  %��batch_size����ȡ��
    ed = min(t+batch_size-1, size(imdb.labels, 2)); 
    [data, labels] = batchFunc(imdb, t:ed); %��batch_size��������ȡ�ļ��б��е�����
    data = gpuArray(data);   %������ת��Ϊgpu���ݣ��������ݴ�MATLAB�����ռ䴫�͵�GPU��
    res = vl_simplenn(net, data);
    rex = squeeze(gather(res(end).x));%res(end).x ��ʾ���һ�㣬gatherΪ��GPU�ϵ�����ת����CPU���ݣ�squeezeΪɾ����һά��

    U(:, t:ed) = rex;
    H(:, t:ed) = single(sign(rex) > 0);   %�����ݱ�Ϊ���������ݡ�
    HLabels(:, t:ed) = labels;
    if verbose && toc > 100
        myLogInfo('%6d / %d', t, size(imdb.labels, 2)); 
        tic;
    end
end
if verbose, toc(t0); end
end
