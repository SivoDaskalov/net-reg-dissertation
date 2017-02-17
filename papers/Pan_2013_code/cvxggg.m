%% Lagrange version
 function[Sbeta]=cvxggg(Y,X,x,del1,del2,tau1,tau2,netwk,wt)
  
  c1=sum(min(abs(x)/tau1,1));
  c2=sum(abs(min( (abs(x(netwk(:,1)))./wt(netwk(:,1))) /tau2,1) - min((abs(x(netwk(:,2)))./wt(netwk(:,2)))/tau2,1)));
  Sbeta=(0.5*square_pos(norm(Y-X*x))+del1*tau1*c1+del2*tau2*c2);
   %Note lam1/tau1=del1, lam2/tau2=del2
 end

