function[x1]=cvxmainalgo(Y,X,del1,del2,tau1,tau2,b0,netwk,wt)
 
   p=size(X,2); 
   
   %1st constraint
   lt=abs(b0)<=tau1;
     
   %2nd constraint
   lgt=abs(b0)./wt> tau2;
   S2=sign(b0).*(1+lgt);
 
    cvx_begin quiet
         variable x(p);
          minimize (0.5*square_pos(norm(Y-X*x))+del1*lt'*abs(x)+2*del2*sum(max(max(abs(x(netwk(:,1)))./wt(netwk(:,1)),abs(x(netwk(:,2)))./wt(netwk(:,2))), abs(x(netwk(:,1)))./wt(netwk(:,1))+abs(x(netwk(:,2)))./wt(netwk(:,2))-tau2))-del2*(S2(netwk(:,1))'*(x(netwk(:,1))./wt(netwk(:,1)))+S2(netwk(:,2))'*(x(netwk(:,2))./wt(netwk(:,2)))));
    cvx_end
    x1=x;   
   
 end
 
