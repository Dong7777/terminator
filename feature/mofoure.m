[hf, full]=fastaread('non_terminators.csv');
Np=length(hf);
positive=full(1:Np);
PPT2=zeros(Np,256); 
AA='ACGT';
for m=1:Np
     M=length(positive{1,m});
    for j=1:M-3
        t1=positive{1,m}(j);
        k1=strfind(AA,t1);
         t2=positive{1,m}(j+1);
        k2=strfind(AA,t2);
         t3=positive{1,m}(j+2);
        k3=strfind(AA,t3);
          t4=positive{1,m}(j+3);
        k4=strfind(AA,t4);
          PPT2(m,k4+4*(k3-1)+16*(k2-1)+64*(k1-1))=PPT2(m,k4+4*(k3-1)+16*(k2-1)+64*(k1-1))+1;

    end
      PPT2(m,:)=PPT2(m,:)/(M-3);
end
          