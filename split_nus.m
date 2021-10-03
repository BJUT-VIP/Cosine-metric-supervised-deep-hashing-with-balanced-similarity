function S = split_nus(Y)

trainPerCls = 10000; testPerCls = 5000;
[N, L] = size(Y);  assert(L == 21);
S = 2 * ones(N, 1);   % default: val
ind = randperm(N);
itest = ind(1:testPerCls);
itrain = ind(testPerCls+1:trainPerCls+testPerCls);
S(itest)=3;
S(itrain)=1;
% chosen = false(N, 1);
% for c = 1:21
%     % use the first testPerCls for test, next trainPerCls for train
%     % but if trainPerCls<=0, use the rest for train
%     ind = find(Y(:,c)>0 & ~chosen);
%     ind = ind(randperm(length(ind)));
%     assert(length(ind) >= trainPerCls + testPerCls);
% 
%     itest = ind(1:testPerCls);
%     if trainPerCls > 0
%         itrain = ind(testPerCls+1:trainPerCls+testPerCls);
%     else
%         % Fatih's bugfix
%         % itrain = ind(testPerCls+1:end);
%         itrain = [];
%     end
%     S(itest) = 3;
%     S(itrain) = 1;
%     chosen([itest; itrain]) = true;
% end
% Fatih's bugfix
% if split == 2
%     S(S == 2) = 1;
% end
end
