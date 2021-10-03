function [topmap] = calPreHn2(Wtrue,HDist,hthreshold)
%WtrueΪ���ƶȾ���HDistΪ���ƶȾ��룬hthresholdΪ�����뾶ֵ
ts = tic();
querynum = size(Wtrue, 1); %��ѯͼ������
testnum = size(Wtrue, 2);  %���ݿ�����
topkap = zeros(1, querynum);
[HDistSorted,HRank] = sort(HDist,2);  %�������򣬴�С���󣬲����HDist2Ϊ��������㹫ʽ
WtrueSorted = zeros(querynum,testnum)>0;
for i=1:querynum
  WtrueSorted(i,:) = Wtrue(i,HRank(i,:));  %������ı�ǩ��ֵ��ȥ��WtrueSortedΪ�ѱ�ǩ��ʵ�ʾ����С������
end
fprintf('sort and ranked down, time is %.3fs\n',toc(ts));
  dH = hthreshold;
  boder = sum(HDistSorted<=dH,2);  %����ÿ��С�ڵ��ں����뾶Ϊ2������
%   quan = sum(HDistSorted, 2);
%   recall = mean(boder./quan);
  for i = 1:querynum
  	gnd = WtrueSorted(i, :); %������ͼƬ�ĺ����ݿ�����ƶȾ�����ȡ
    tgnd = gnd(1:boder(i));  %ȡ�����뾶Ϊ2�ı�ǩ
    if sum(tgnd) == 0  %���û�к����뾶Ϊ2�����ݣ�������
        continue;
    end
    tcount = 1:sum(tgnd);  %�����ۼ�ֵ
    tindex = find(tgnd == 1); %�ҵ����������ǩС�ڵ���2�ı�ǩ
    topkap(i)=mean(tcount./tindex);
  end
  topmap = mean(topkap, 2);
end