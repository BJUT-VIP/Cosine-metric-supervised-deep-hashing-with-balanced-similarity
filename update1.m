function gpu_net = update1(gpu_net, res_back, lr, scratch)
    weight_decay = 5*1e-4;
    n_layers = 20 ;
    batch_size = 64;
    for ii = 1:n_layers
        if ii <20
            if ~isempty(gpu_net.layers{ii}.weights)
                    gpu_net.layers{ii}.weights{1} = gpu_net.layers{ii}.weights{1}+...
                        lr*(res_back(ii).dzdw{1}/(batch_size*batch_size) - weight_decay*gpu_net.layers{ii}.weights{1});
                    gpu_net.layers{ii}.weights{2} = gpu_net.layers{ii}.weights{2}+...
                        lr*(res_back(ii).dzdw{2}/(batch_size*batch_size) - weight_decay*gpu_net.layers{ii}.weights{2});
            end
        elseif ii == 20
            if ~isempty(gpu_net.layers{ii}.weights)
                    gpu_net.layers{ii}.weights{1} = gpu_net.layers{ii}.weights{1}+...
                        scratch * lr *(res_back(ii).dzdw{1}/(batch_size*batch_size) - weight_decay*gpu_net.layers{ii}.weights{1});
                    gpu_net.layers{ii}.weights{2} = gpu_net.layers{ii}.weights{2}+...
                        scratch *lr *(res_back(ii).dzdw{2}/(batch_size*batch_size) - weight_decay*gpu_net.layers{ii}.weights{2});
            end
            
        end
    end
end