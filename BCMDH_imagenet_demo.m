function [Loss, Loss1, Loss2, map] = BCMDH_imagenet_demo(codelens, eta, learningrate)
    %% setting seeds
    rand('seed', 0);
    %% download dataset or 
    %% load the Dataset
%      load('E:/BCMDH_NDCG\plot_pr_map/euclidean_imagenet/cos_100_64_0.0025_90_.mat')
    net = load('E:/TALR-master/data/imagenet-caffe-alex.mat');
    %% initialization
    maxIter = 100;
%     lr = logspace(-1, -3, maxIter);
    lr = learningrate;
    net = net_structure(net, codelens);
    

%     global IMDBB
    images = {};
    name = 'imagenet';
    images = get_imdb(images, net, name);
    disp(images)
    
    imgSize = net.meta.normalization.imageSize(1);   %Í¼ï¿½ï¿½ß´ï¿?
    meanImage = single(net.meta.normalization.averageImage); %ï¿½ï¿½ï¿½ï¿½Í¼ï¿½ï¿½Æ½ï¿½ï¿½
    if isequal(size(meanImage), [1 1 3]) 
        meanImage = repmat(meanImage, [imgSize imgSize]);  %repmatï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½meanimageï¿½ï¿½ï¿½ï¿½imgsize*imagsize
    else
        assert(isequal(size(meanImage), [imgSize imgSize 3]));
    end
    testbatchFunc = @(I, B) batch_imagenet(I, B, imgSize, meanImage);%IÎªimdbï¿½æ´¢ï¿½ï¿½txtï¿½ï¿½ï¿½Ý£ï¿½BÎªbatch_sizeï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½È¡ï¿½ï¿½ï¿½ï¿½
    trainbatchFunc = @getSimpleNNBatch;

    U = zeros(size(images.train.labels, 2), codelens);
    B = zeros(size(images.train.labels, 2), codelens);
    label = zeros(size(images.train.labels, 2), 1);
    Loss = zeros(1, maxIter);
    Loss1 = zeros(1, maxIter);
    Loss2 = zeros(1, maxIter);
    map = zeros(1, maxIter);
    top = 1000;
    scratch = 50; % As the fch layer, we set its learning rates for the imagenet datasets as 50 times that of the lower layers.
    %% training
    for iter = 1: maxIter
        [net, loss, loss1, loss2, U, B, label] = BCMDH_loss(images.train, trainbatchFunc, net, iter, lr, eta, U, B, label, scratch);
        Loss(1, iter) = loss;
        Loss1(1, iter) = loss1;
        Loss2(1, iter) = loss2;
        
        if (iter ==1 || iter == 10)
            Y = tsne(U, 'Algorithm','barneshut', 'NumPCAComponents',codelens);
            figure('name', 'ÊµÊýÖµ')
            gscatter(Y(:,1),Y(:,2), label)
%         
% %             Y3D = tsne(U, 'Algorithm','barneshut', 'NumPCAComponents',codelens,'NumDimensions',3);
% %             figure('name', '3DÊµÊýÖµ')
% %             scatter3(Y3D(:,1),Y3D(:,2),Y3D(:,3),15,label,'filled');
% %             view(-93,14)
            Y1 = tsne(B, 'Algorithm','barneshut');
        
            figure('name', '¹þÏ£Öµ')
            gscatter(Y1(:,1),Y1(:,2), label)
        
%             YB3D = tsne(B, 'Algorithm','barneshut','NumDimensions',3);
%             figure('name', '3D¹þÏ£Öµ')
%             scatter3(YB3D(:,1),YB3D(:,2),YB3D(:,3),15,label,'filled');
%             view(-93,14)
        end
        if mod(iter, 80) == 0
            lr = lr * 0.1;
        end     
    end
    save(['E:/BCMDH_imagenet_' num2str(eta) '_' num2str(codelens) '_' num2str(lr) '_' num2str(maxIter) '_.mat'],'net')
    %% testing
    [U_text, B_test, testL,U_dataset, B_dataset, retrievalL,topkmap, precision, HR, MHam2pre,ndcg] = test_evaluation(net, images.test, images.database, testbatchFunc, top, codelens);
    disp(['Top-1K mAP:']);
    disp(topkmap);
    disp(ndcg);
    save(['BCMDH_Imagenet_' num2str(codelens) '_' num2str(eta) '_' num2str(learningrate) '_' num2str(alpha) 'bits.mat'],'U_text', 'B_test', 'testL','U_dataset', 'B_dataset', 'retrievalL','topkmap', 'precision', 'HR', 'MHam2pre','ndcg');
end