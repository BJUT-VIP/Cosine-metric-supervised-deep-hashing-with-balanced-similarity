function [U_text, B_test, testL,U_dataset, B_dataset, retrievalL,topkmap, precision, HR, MHam2pre,ndcg] = test_evaluation(net, test, database, batchFunc, top, codelens)
topk = top;
pos = [100:100:1000];
% test_id = randperm(numel(test.data))
% database_id = randperm(numel(database.data))
[U_text, B_test, testL] = hash_encode(net, batchFunc, test, codelens);  %codelens*数据库长度，label：类别*数据库长度
[U_dataset, B_dataset, retrievalL] = hash_encode(net, batchFunc, database, codelens);

B_dataset1 = compactbit(B_dataset'); %数据库长度*nwords
B_test1 = compactbit(B_test');
[S, Aff] = compute_S(retrievalL', testL');
[topkmap] = calculateTopMap(B_dataset1,B_test1,S',testL',topk); %testL:数据库长度*标签类别， retrievalL：标签类别*数据库长度

Dhamm = hammingDist(B_test1, B_dataset1);
[~, precision] = recall_precision5(S', Dhamm, pos);  % the accuracy of top-K retrieved images
HR = callHLLabel(S', Dhamm);  % PR curves
MHam2pre = MH2P(S', Dhamm);  % the precision curves within Hamming distance 2

ndcg = tieNDCG(B_test, B_dataset, Aff);
disp(topkmap);
end