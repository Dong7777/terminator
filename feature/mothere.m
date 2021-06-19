[hf, full]=fastaread('non_terminators.csv');
Np=length(hf);
positive=full(1:Np);
PPT1=zeros(Np,64); 
AA='ACGT';
for m=1:Np
     M=length(positive{1,m});
    for j=1:M-2
        t1=positive{1,m}(j);
        k1=strfind(AA,t1);
         t2=positive{1,m}(j+1);
        k2=strfind(AA,t2);
         t3=positive{1,m}(j+2);
        k3=strfind(AA,t3);
        PPT1(m,k3+4*(k2-1)+16*(k1-1))=PPT1(m,k3+4*(k2-1)+16*(k1-1))+1;
    end
      PPT1(m,:)=PPT1(m,:)/(M-2);
end
