 function[b]=lasso(Y,X,lam)
 
 p=size(X,2);
  
    cvx_begin quiet
      variable x(p);
      minimize (0.5*square_pos(norm(Y-X*x)));
      subject to 
       sum(abs(x)) <= lam;
    cvx_end
    b=x;
  
 end    
