function result = callHLLabel(Wtrue, Dhamm)

% Dhamm = hammingDist(tB, dB);
% Wtrue = calcGnd(testL, databaseL);

maxHamm = max(Dhamm(:));
totalGoodPairs = sum(Wtrue(:));

% find pairs with similar codes
precision = zeros(maxHamm, 1);
recall = zeros(maxHamm, 1);

for n = 1: length(precision)
    j = (Dhamm <= ((n-1) + 0.00001));
% j = (Dhamm <= (1 + 0.00001));
    retrievalGoodPairs = sum(Wtrue(j));
    
    retrievalPairs = sum(j(:));
    precision(n) = retrievalGoodPairs / (retrievalPairs + eps);
    recall(n) = retrievalGoodPairs / totalGoodPairs;
%     precision = retrievalGoodPairs / (retrievalPairs + eps);
%     recall = retrievalGoodPairs / totalGoodPairs;
% end

result.pre = precision;
result.rec = recall;

end