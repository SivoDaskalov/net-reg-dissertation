function[b1]=cvxtip(Y,X,del1,del2,tau1,tau2,b0,netwk,wt)
 
  b1=cvxmainalgo(Y,X,del1,del2,tau1,tau2,b0,netwk,wt);
  gb1=cvxggg(Y,X,b1,del1,del2,tau1,tau2,netwk,wt);
       
    b2=cvxmainalgo(Y,X,del1,del2,tau1,tau2,b1,netwk,wt);
    gb2=cvxggg(Y,X,b2,del1,del2,tau1,tau2,netwk,wt);

     while (gb1-gb2>0.001)
      gb1=gb2;
      b1=b2;
      b2=cvxmainalgo(Y,X,del1,del2,tau1,tau2,b1,netwk,wt);
      gb2=cvxggg(Y,X,b2,del1,del2,tau1,tau2,netwk,wt);
     end

end
