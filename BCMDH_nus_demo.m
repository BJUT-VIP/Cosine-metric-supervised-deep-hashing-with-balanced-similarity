function [Loss, Loss1, Loss2, map] = BCMDH_nus_demo(codelens, eta, learningrate)
    %% setting seeds
    seed=0;
    rng('default');
    rng(seed);
%     rand('seed', 0);
    %% download dataset or 
    %% load the Dataset
    net = load('E:/TALR-master/data/imagenet-caffe-alex.mat');
%     net = load('imagenet-vgg-f.mat');
    %% initialization
    maxIter = 60;
%     lr = logspace(-1, -3, maxIter);
    lr = learningrate;
    net = net_structure(net, codelens);
    

%   global IMDBB
    images = {};
    name = 'nus';
    images = get_imdb(images, net, name);
    disp(images)
    
    imgSize = net.meta.normalization.imageSize(1);   %ͼ��ߴ???
    meanImage = single(net.meta.normalization.averageImage); %����ͼ��ƽ��
    if isequal(size(meanImage), [1 1 3]) 
        meanImage = repmat(meanImage, [imgSize imgSize]);  %repmat��������meanimage����imgsize*imagsize
    else
        assert(isequal(size(meanImage), [imgSize imgSize 3]));
    end
    testbatchFunc = @(I, B) batch_imagenet(I, B, imgSize, meanImage);%IΪimdb�洢��txt���ݣ�BΪbatch_size����������ȡ����
%     if strcmp(name, 'nus')
%         trainbatchFunc =@(I, B) trainbatch_imagenet(I, B, imgSize, meanImage);
%     else
    trainbatchFunc = @getSimpleNNBatch;
%     end
%     data_train = [];
%     if isempty(data_train), data_train = find(IMDB.images.set==1) ; end %��ȡѵ�����ݣ����??1
%     if isnan(data_train), data_train = [] ; end  %�ж�����Ԫ���Ƿ����NaN����ΪNaN���ÿգ���ֹ����ѵ������
    
    U = zeros(size(images.train.labels, 2), codelens);
    B = zeros(size(images.train.labels, 2), codelens);
    label = zeros(size(images.train.labels, 2), 1);

    Loss = zeros(1, maxIter);
    Loss1 = zeros(1, maxIter);
    Loss2 = zeros(1, maxIter);
    map = zeros(1, maxIter);
    top = 5000;
    scratch = 10; % As the fch layer, we set its learning rates for the nus-wide datasets as 10 times that of the lower layers.
    %% training
    for iter = 1: maxIter
        [net, loss, loss1, loss2, U, B, label] = BCMDH_loss(images.train, trainbatchFunc, net, iter, lr, eta, U, B, label, scratch);
        Loss(1, iter) = loss;
        Loss1(1, iter) = loss1;
        Loss2(1, iter) = loss2;
        
        if (iter ==1 || iter == 10)
            Y = tsne(U, 'Algorithm','barneshut', 'NumPCAComponents',codelens);
            figure('name', 'ʵ??ֵ')
            gscatter(Y(:,1),Y(:,2), label)
%         
% %             Y3D = tsne(U, 'Algorithm','barneshut', 'NumPCAComponents',codelens,'NumDimensions',3);
% %             figure('name', '3Dʵ??ֵ')
% %             scatter3(Y3D(:,1),Y3D(:,2),Y3D(:,3),15,label,'filled');
% %             view(-93,14)
            Y1 = tsne(B, 'Algorithm','barneshut');
        
            figure('name', '??ϣֵ')
            gscatter(Y1(:,1),Y1(:,2), label)
        
%             YB3D = tsne(B, 'Algorithm','barneshut','NumDimensions',3);
%             figure('name', '3D??ϣֵ')
%             scatter3(YB3D(:,1),YB3D(:,2),YB3D(:,3),15,label,'filled');
%             view(-93,14)
        end
        if mod(iter, 40) == 0
            lr = lr * 0.1;
        end     
    end
    save(['E:/BCMDH_NUS_' num2str(eta) '_' num2str(codelens) '_' num2str(lr) '_' num2str(maxIter) '_.mat'],'net')
%     load('BCMDH_NUS_WIDE_40_.mat')
    %% testing
    [U_text, B_test, testL,U_dataset, B_dataset, retrievalL,topkmap, precision, HR, MHam2pre,ndcg] = test_evaluation(net, images.test, images.database, testbatchFunc, top, codelens);
    disp(['Top-5K mAP:']);
    disp(topkmap);
    disp(ndcg);
    save(['BCMDH_NUS_' num2str(codelens) '_' num2str(eta) '_' num2str(learningrate) '_' num2str(alpha) 'bits.mat'],'U_text', 'B_test', 'testL','U_dataset', 'B_dataset', 'retrievalL','topkmap', 'precision', 'HR', 'MHam2pre','ndcg');
end