clc
clear
load('mer5p.mat')
load('mer5n.mat')
train_y = [1*ones(1,size(mer5p,1)),-1*ones(1,size(mer5n,1))]';%类标
train=[mer5p;mer5n];
s=size(train,2);%特征数
s1=size(train,1);%正负样本总数

%计算max-relevance
X=zeros(1,s);%相关系数赋初值
for i=1:s
xishu=corr(train(:,i),train_y,'type','Pearson');
X(1,i)=xishu;
end
% 计算max-distance  欧氏距离
ED=pdist2(train',train','euclidean');

%计算最大距离
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
PositiveBP=zeros(280,1024);%1484*126，提取了126个特征
NegativeBP=zeros(560,1024);%1484*126
for i=1:1024
    k=rank_ed(2,i);%F第二行第i个数
    PositiveBP(:,i)=mer5p(:,k);%1484*126，x1的第k列称为 PositiveBP的第i列
    NegativeBP(:,i)=mer5n(:,k);
end

