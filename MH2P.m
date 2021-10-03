function result = MH2P(Wtrue, Dhamm)


j = (Dhamm <= (2 + 0.00001));
    retrievalGoodPairs = sum(Wtrue.*j,2);
    
    retrievalPairs = sum(j,2);
    precision = retrievalGoodPairs ./ (retrievalPairs + eps);
%     recall = sum(retrievalGoodPairs(:)) /(sum(j(:)));
%     recall1 = sum(retrievalGoodPairs(:))/ (sum(Dhamm(:)));
result = mean(precision);


end