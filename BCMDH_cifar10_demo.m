function [Loss, Loss1, Loss2, map] = BCMDH_cifar10_demo(codelens, eta, learningrate,alpha)
    %% setting seeds
    seed = 0;
    rng('default');
    rng(seed);
    if exist('cifar-10.mat', 'file')
        load('/data2/hwj/old_paper_code/CMDH_cifar1/cifar-10.mat');
    else
        data_prepare;
        load('/data2/hwj/old_paper_code/CMDH_cifar1/cifar-10.mat');
    end
%     rand('seed', 0);
    %% download dataset or 
    %% load the Dataset
    net = load('E:/TALR-master/data/imagenet-caffe-alex.mat');
    %% initialization
    maxIter = 50;
    lr = learningrate;
    net = net_structure(net, codelens);
    

%     global IMDBB 
    U = zeros(size(train_data,4),codelens); %Ñµï¿½ï¿½ï¿½ï¿½5000*bitï¿½ï¿½ï¿½ï¿½Ï£ï¿½ï¿½Êµï¿½ï¿½Öµ
    B = zeros(size(train_data,4),codelens); %Ñµï¿½ï¿½ï¿½ï¿½5000*bitï¿½ï¿½ï¿½ï¿½Ï£ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Öµ
    label = zeros(size(train_L, 2), 1);
    Loss = zeros(1, maxIter);
    Loss1 = zeros(1, maxIter);
    Loss2 = zeros(1, maxIter);
    map = zeros(1, maxIter);
    top = 54000;
    scratch = 5; % As the fch layer, we set its learning rates for the cifar10 datasets as 5 times that of the lower layers.
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
        if mod(iter, 30) == 0
            lr = lr * 0.1;
        end     
    end
    save(['E:/BCMDH_CIFAR10_' num2str(eta) '_' num2str(codelens) '_' num2str(lr) '_' num2str(maxIter) '_.mat'],'net')
    %% testing
    [U_text, B_test, testL,U_dataset, B_dataset, retrievalL,topkmap, precision, HR, MHam2pre,ndcg] = test_evaluation(net, images.test, images.database, testbatchFunc, top, codelens);
    disp(['Top-54K mAP:']);
    disp(topkmap);
    disp(ndcg);
    save(['BCMDH_CIFAR10_' num2str(codelens) '_' num2str(eta) '_' num2str(learningrate) '_' num2str(alpha) 'bits.mat'],'U_text', 'B_test', 'testL','U_dataset', 'B_dataset', 'retrievalL','topkmap', 'precision', 'HR', 'MHam2pre','ndcg');
end