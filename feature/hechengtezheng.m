clc
clear

load('EIIP425.mat')
load('EIIP.mat')
load('1.mat')
load('2.mat')
positive836=positive147(:,1:4032);
negative836=positive425(:,1:4032);
dc4251=[negative836,EIIP425];
dc1471=[positive836,EIIp];