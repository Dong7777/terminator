[hf, full]=fastaread('terminators.csv');
Np=length(hf);
positive=full(1:Np);
PPT2=zeros(Np,64); 
AA='ACGT';
EIIP_ACGT=[0.126 0.134 0.0806 0.1335];
for m=1:Np
     M=length(positive{1,m});
    for j=1:M-2
        t1=positive{1,m}(j);
        k1=strfind(AA,t1);
         t2=positive{1,m}(j+1);
        k2=strfind(AA,t2);
         t3=positive{1,m}(j+2);
        k3=strfind(AA,t3);
        PPT2(m,k3+4*(k2-1)+16*(k1-1))=EIIP_ACGT(1,k1)+EIIP_ACGT(1,k2)+EIIP_ACGT(1,k3);
    end
  
end
PPT4=mer3p.*PPT2;
