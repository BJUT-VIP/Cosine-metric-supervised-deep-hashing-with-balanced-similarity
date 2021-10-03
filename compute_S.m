function [S, Aff] = compute_S (train_L,test_L)
%     train_L = single(train_L) ;
%     test_L = single(test_L) ;
%     Dp = repmat(train_L,1,length(test_L)) - repmat(test_L',length(train_L),1);
%     S = Dp == 0;
if size(test_L, 2) == 1
    train_L = single(train_L) ;
    test_L = single(test_L) ;
    Dp = repmat(train_L,1,length(test_L)) - repmat(test_L',length(train_L),1);
    S = Dp == 0;
elseif size(test_L, 2) > 1
%     L1 = single(train_L);
%     L2 = single(test_L);
    D = train_L * test_L' ;
    S = D >0;
    Aff = D';
end
end