[hf, full]=fastaread('non_terminators.csv');
Np=length(hf);
positive=full(1:Np);
PPT2=zeros(Np,4); 
AA='ACGT';
for i=1:Np
       L=length(positive{1,i});
    for j=1:L
        t1=positive{1,i}(j);
        k1=strfind(AA,t1);
        PPT2(i,k1)=PPT2(i,k1)+1;
    end
   PPT2(i,:)=PPT2(i,:)/L;
end