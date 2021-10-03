% calculate a logic matrix indicating whether a pair of items are neighbors.
function [S, R, spw, snw] = similarity_label(label)
        
    DR = label' * label;
    S = DR > 0;
    sp = sum(S(:));
    t = 1 - S;
    sn = sum(t(:));
    st = sn + sp;
    spw = fix(st / sp);
    snw = fix(st/ sn);
    mD = max(DR, [], 2);
    R = DR ./ mD;
    R(find(R == Inf | isnan(R))) = 0;
end
