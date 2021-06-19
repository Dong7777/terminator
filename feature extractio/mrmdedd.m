clc
clear
load('mer5p.mat')
load('mer5n.mat')
train_y = [1*ones(1,size(mer5p,1)),-1*ones(1,size(mer5n,1))]';%���
train=[mer5p;mer5n];
s=size(train,2);%������
s1=size(train,1);%������������

%����max-relevance
X=zeros(1,s);%���ϵ������ֵ
for i=1:s
xishu=corr(train(:,i),train_y,'type','Pearson');
X(1,i)=xishu;
end
% ����max-distance  ŷ�Ͼ���
ED=pdist2(train',train','euclidean');

%����������
E=zeros(1,s);
for i=1:s
E(i)=1/(s-1)*(sum(ED(i,:)));
end
maxed=max(E);

for i=1:s
    if isnan(ED(1,i))
        ED(1,i)=0;
    end
end

maxmax=X+E;
[sort1, sort2]=sort(maxmax,'descend');
rank_ed=[sort1;sort2];
save rank_ed rank_ed
PositiveBP=zeros(280,1024);%1484*126����ȡ��126������
NegativeBP=zeros(560,1024);%1484*126
for i=1:1024
    k=rank_ed(2,i);%F�ڶ��е�i����
    PositiveBP(:,i)=mer5p(:,k);%1484*126��x1�ĵ�k�г�Ϊ PositiveBP�ĵ�i��
    NegativeBP(:,i)=mer5n(:,k);
end

