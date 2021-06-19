[hf, full]=fastaread('non_terminators.csv');
Np=length(hf);
positive=full(1:Np);
PPT2=zeros(Np,4096); 
AA='ACGT';
for m=1:Np
     M=length(positive{1,m});
    for j=1:M-5
        t1=positive{1,m}(j);
        k1=strfind(AA,t1);
         t2=positive{1,m}(j+1);
        k2=strfind(AA,t2);
         t3=positive{1,m}(j+2);
        k3=strfind(AA,t3);
          t4=positive{1,m}(j+3);
        k4=strfind(AA,t4);
         t5=positive{1,m}(j+4);
        k5=strfind(AA,t5);
         t6=positive{1,m}(j+5);
        k6=strfind(AA,t6);
         PPT2(m,k6+4*(k5-1)+16*(k4-1)+64*(k3-1)+256*(k2-1)+1024*(k1-1))=PPT2(m,k6+4*(k5-1)+16*(k4-1)+64*(k3-1)+256*(k2-1)+1024*(k1-1))+1;
    end
      PPT2(m,:)=PPT2(m,:)/(M-5);
end

          
         