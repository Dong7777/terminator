  clc
clear
load('mer5p.mat')
load('mer5n.mat')
x1=mer5p;
x2=mer5n;

% x1��ʾ����������% x2��ʾ����������
x = [x1',x2']';
% xΪ�ܵ�����
[n1,m1] = size(x1);
[n2,m2] = size(x2);
%�ҳ����������Ӧ��ά�ȣ�n��������m��ά��
aver1 = mean(x1);
aver2 = mean(x2);
aver3 =mean(x);
%����������ĸ�������ƽ��ֵ����Ϊ1*m������
numrator = (aver1-aver3).^2+(aver2-aver3).^2;
%��Ϊ������ӣ��õ��Ļ���һ��1*m������
sum_1 = zeros(1,m1);
%����ֵ�����ܼ������
for k=1:n1
    chazhi_1 = x1(k,:)-aver1;
    added_1 = chazhi_1 .^2;
    sum_1 = sum_1 + added_1;
end
deno_1 = sum_1/(n1 - 1);
%��forѭ�����ˣ��õ��˷�ĸ��ǰ�벿��
sum_2 = zeros(1,m2);
for k = 1:n2
    chazhi_2 = x2(k,:)-aver2;
    added_2 = chazhi_2 .^2;
    sum_2 = sum_2 +added_2;
end
deno_2 = sum_2/(n2 - 1);
%�õ���ĸ�ĺ�벿��
deno = deno_1 + deno_2;
%�õ���ĸ
F_1 = numrator ./ deno;
%�õ�����δ���������F

len = length(F_1);
for k = 1:len
    if isnan(F_1(k))
        F_1(k) = -1;
    end
end
% ȥ����F_1��ֵΪNAN��ֵ
[F_2,ind] = sort(F_1,'descend');
% ����F_1�����˽������У�ind����index
F = [F_2',ind']';


PositiveBP=zeros(n1,836);%1484*126����ȡ��126������
NegativeBP=zeros(n2,836);%1484*126
for i=836
    k=F(2,i);%F�ڶ��е�i����
    PositiveBP(:,i)=x1(:,k);%1484*126��x1�ĵ�k�г�Ϊ PositiveBP�ĵ�i��
    NegativeBP(:,i)=x2(:,k);
end
