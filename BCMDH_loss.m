function [net, loss, loss1_1, loss2_1, U1, B1, tlabel] = BCMDH_loss(train_data, batchFunc, net, iter ,lr,eta, U1, B1, tlabel, scratch) %���ݷֱ�Ϊѵ�����ݡ���ǩ����ϣ��ʵ������ϣֵ������ṹ��ѧϰ�ʡ�������?
    N = size(train_data.labels, 2);
    data_train = randperm(N);  
    batchsize = 64;   
    myLogInfo('Epoch %d, learningRate = %d, eta = %d', iter, lr, eta);
    for t=0:ceil(N/batchsize)-1
        batch_time=tic ;     % start
        batch = data_train((1+t*batchsize):min((t+1)*batchsize,N));
        [im_, labels] = batchFunc(train_data, batch) ; %labels:21*batchsize or 80*batchsize
        [r, c] = find(labels');
        labels1 = [];
        labels1(r) = c;
        %% Calculate the similarity matrix
        [S, R, spw, snw] = similarity_label(labels) ;  % calculate the similarity matrix, similarity degree, weight
        %% load and preprocess an image
        im_ = gpuArray(im_) ;  % images 224*224*3*batchsize
        %% run the CNN
        res = vl_simplenn(net, im_) ; % CNN network 
        U0 = squeeze(gather(res(end).x))' ; % the output of the last layer of the CNN network
        U  = U0 ;  %����Ӧ�Ĺ�ϣ��ʵ�����뵽��U��
        U1(batch, :) = U0;
        B1(batch, :) = sign(U0);
        tlabel(batch, :) = labels1';
        batchB= sign(U0);  % hash codes 
        
%% Calculate the cosine distance
        dot = U0 * U';
        u0_norm = sum(abs(U0).^2, 2).^(1/2);
        u_norm = sum(abs(U).^2, 2).^(1/2);
        norm_dot = u0_norm * u_norm';
        norm_dot(norm_dot == 0) = 1e-6;
        cosine_similarity = dot ./ norm_dot;
        cosine_distance = 1 - cosine_similarity;
        cosine_distance(cosine_distance <= 0.0) = 1e-7;
        cosine_kong = 1 + cosine_similarity;
        cosine_kong(cosine_kong<= 0.0) = 1e-7;
                
%% Calculate the cosine distance entropy quantization loss and total loss
        cos_dot = sum(U0 .* batchB, 2);
        batch_norm = sum(abs(batchB).^2, 2).^(1/2);
        cosnorm_dot = u0_norm .* batch_norm;
        cos_regular = cos_dot ./ cosnorm_dot;
        cos_similarity = 1 + cos_regular;
        log_cos_regular = log(0.5*cos_similarity);
        m = 0.0;  % margin threshold 
        cosine_similarity1 = cosine_similarity - m;
        panduan = cosine_similarity >m;

        log_poscosine = log(0.5 * cosine_kong);
        log_negcosine = log(0.5 * cosine_distance);
        loss_pos = R.* spw .* S .* (cosine_distance).^(2);
        loss_neg = snw.*(1-S).* panduan.* cosine_similarity1.^(2);
        
        loss1 = -loss_pos .* log_poscosine - loss_neg .* log_negcosine;
        loss2 = -eta * (1 - cos_regular).*log_cos_regular;

        loss1_1 = mean(loss1(:));
        loss2_1 = mean(loss2(:));
        loss = loss1_1 + loss2_1;
        
       %% For calculation back-propagation
        cos_pos = 2 * R.* spw .*S .* (cosine_distance);
        cos_neg = 2*snw.*(1-S).* panduan.*(cosine_similarity1);
        
        dis_S = cos_pos .* log_poscosine;
        p_norm_dot = dis_S ./ norm_dot;
        part_1 = p_norm_dot * U;
        u01_norm = u0_norm.^3;
        norm_dot1 = u01_norm * u_norm';
        norm_dot1(norm_dot1 == 0) = 1e-6;
        snorm_dot1 = dis_S ./ norm_dot1;
        part_2 = sum(dot .* snorm_dot1, 2);
        dis_ns = cos_neg .*log_negcosine ;
        p_norm_dot1 = dis_ns ./ norm_dot;
        npart_1 = p_norm_dot1 * U;
        snorm_dot2 = dis_ns ./ norm_dot1;
        npart_2 = sum(dot .* snorm_dot2, 2);
        cosine_grad = ((part_1 - part_2 .* U0) - (npart_1 - npart_2 .* U0));
        
        log_p_norm_dot = loss_pos ./ (norm_dot .* cosine_kong);
        log_part_1 = log_p_norm_dot * U;
        log_snorm_dot1 = loss_pos ./(norm_dot1 .* cosine_kong);
        log_part_2 = sum(dot .* log_snorm_dot1, 2);
        log_posgrad = log_part_1 - log_part_2 .* U0;
       
        log_p_norm_dot1 = loss_neg ./ (norm_dot .* cosine_distance);
        log_part_11 = log_p_norm_dot1 * U;
        log_snorm_dot11 = loss_neg ./(norm_dot1 .* cosine_distance);
        log_part_21 = sum(dot .* log_snorm_dot11, 2);
        log_neggrad = log_part_11 - log_part_21 .* U0;

       %% To calculate the feedback of the cosine distance entropy quantization loss
        cos_part1 = batchB./ cosnorm_dot;
        cos_part2 = (cos_dot .* U0) ./ (u01_norm .* batch_norm);
        cos_regular_grad = cos_part1 - cos_part2;
        
        regu1 = cos_regular_grad.* log_cos_regular;
        temp = (1 - cos_regular)./ cos_similarity;
        regu2 = temp .* cos_regular_grad;
        regu_grad = -eta * (regu1 - regu2);

       %% total back-propagation
        dJdU = 2*(- cosine_grad  + log_posgrad - log_neggrad) + regu_grad;           
        dJdoutput = gpuArray(reshape(dJdU',[1,1,size(dJdU',1),size(dJdU',2)])) ;
        res = vl_simplenn(net, im_, dJdoutput) ;
       %% update the parameters of CNN
        net = update1(net , res, lr, scratch) ;
        batch_time = toc(batch_time) ;
        fprintf('iter %d  batch %d/%d (%.1f images/s) ,lr is %d  ', iter, t+1,ceil(N/batchsize), batchsize/ batch_time,lr) ;
        fprintf('loss is %.5f loss1_1 %.5f loss2_1 %.5f\n',loss, loss1_1,loss2_1);
    end
end

