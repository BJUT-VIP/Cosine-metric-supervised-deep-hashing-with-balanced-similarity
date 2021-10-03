function [topmap] = calPreHn2(Wtrue,HDist,hthreshold)
%Wtrue为相似度矩阵，HDist为相似度距离，hthreshold为汉明半径值
ts = tic();
querynum = size(Wtrue, 1); %查询图像数量
testnum = size(Wtrue, 2);  %数据库数量
topkap = zeros(1, querynum);
[HDistSorted,HRank] = sort(HDist,2);  %按行排序，从小到大，查表法，HDist2为汉明距计算公式
WtrueSorted = zeros(querynum,testnum)>0;
for i=1:querynum
  WtrueSorted(i,:) = Wtrue(i,HRank(i,:));  %把排序的标签赋值过去，WtrueSorted为把标签按实际距离从小到大排
end
fprintf('sort and ranked down, time is %.3fs\n',toc(ts));
  dH = hthreshold;
  boder = sum(HDistSorted<=dH,2);  %计算每行小于等于汉明半径为2的数量
%   quan = sum(HDistSorted, 2);
%   recall = mean(boder./quan);
  for i = 1:querynum
  	gnd = WtrueSorted(i, :); %把这张图片的和数据库的相似度矩阵提取
    tgnd = gnd(1:boder(i));  %取汉明半径为2的标签
    if sum(tgnd) == 0  %如果没有汉明半径为2的数据，则跳过
        continue;
    end
    tcount = 1:sum(tgnd);  %计算累计值
    tindex = find(tgnd == 1); %找到汉明距离标签小于等于2的标签
    topkap(i)=mean(tcount./tindex);
  end
  topmap = mean(topkap, 2);
end