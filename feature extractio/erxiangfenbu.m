clc
clear
load('C:\Users\hp\Desktop\6-mer\二项分布\n61.mat');
load('C:\Users\hp\Desktop\6-mer\二项分布\p61.mat');
a1=sum(p6);
b=sum(a1);
a2=sum(n6);
m=sum(a2);
na=b+m;
z=[];
z(1,:)=a1;
z(2,:)=a2;
z(3,:)=sum(z);
p=b/(b+m);
a=[];
for j=1:4096
    P=[];
for i=z(1,j)+1:z(3,j)+1
    i1=i-1;
    P(i)=binopdf(i1,z(3,j),p);
end
P1=sum(P);
a(j)=P1;
end
m1=m/(b+m);
c=[];
for j=1:4096
    P=[];
for i=z(2,j)+1:z(3,j)+1
    i1=i-1;
    P(i)=binopdf(i1,z(3,j),m1);
end
P1=sum(P);
c(j)=P1;
end
b=[];
b(1,:)=a;
b(2,:)=c;
[min_a,index]=min(b);
[sort1, sort2]=sort(min_a);
rank=[sort1;sort2];

PositiveBP=zeros(280,4096);%1484*126，提取了126个特征
NegativeBP=zeros(560,4096);%1484*126
for i=1:4096
    k=rank(2,i);%F第二行第i个数
    PositiveBP(:,i)=p6(:,k);%1484*126，x1的第k列称为 PositiveBP的第i列
    NegativeBP(:,i)=n6(:,k);
end