[hf, full]=fastaread('non_terminators.csv');
Np=length(hf);
positive=full(1:Np);
pjh2=zeros(Np,16); 
AA='ACGT';
for m=1:Np
    M=length(positive{1,m});
    for k=1:M-1
        s=positive{1,m}(k);
        t=positive{1,m}(k+1);
        i=strfind(AA,s);
        j=strfind(AA,t);
       pjh2(m,j+(i-1)*4)=pjh2(m,j+(i-1)*4)+1;
    end
      pjh2(m,:)=pjh2(m,:)/(M-1);
end