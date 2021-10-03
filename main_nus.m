addpath E:/MATLAB2018/matconvnet-1.0-beta25/matlab/    %¼ÓÈëËÑË÷Â·¾¶
run E:/VLFeat/vlfeat-0.9.21/toolbox/vl_setup
% run /usr/local/MATLAB/R2017b/matconvnet-1.0-beta25/matlab/vl_setupnn
gpuDevice(1)                                %ï¿½ï¿½ï¿½Ã¼ï¿½ï¿½ï¿½GPU
vl_setupnn;                                 %ï¿½ï¿½ï¿½ï¿½MatConvNetï¿½ï¿½ï¿½ï¿½ï¿½ä£¬ï¿½ï¿½MatConvNetï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Óµï¿½MATLABÂ·ï¿½ï¿½ï¿½ï¿½
for bit = [16, 32, 48, 64]
    for lr = [0.025]
        for  eta = [100]
                [Loss, Loss1, Loss2, map] = BCMDH_nus_demo(bit, eta, lr);
%         figure(1)
%         hold on ;
%         x = 1:50;
%         plot(x, Loss, 'r-*', x, Loss1, 'b--o', x, Loss2, 'k*');
%         legend('Loss', 'cos', 'regurla')
%         figure(2)
%         hold on ;
%         plot(x, map);
        end
    end
end