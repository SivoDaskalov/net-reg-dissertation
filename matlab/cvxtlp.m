function [b, s] = cvxtlp(Y, X, wt, netwk, b0, delta1, delta2, tau1, tau2)
    p=size(X,2);
    %1st constraint
    lt=abs(b0) <= tau1;
    %2nd constraint
    lgt=abs(b0)./wt > tau2;
    S2=sign(b0).*(1+lgt);
    
    cvx_begin quiet
%         cvx_precision(0.999);
%         cvx_solver sedumi;
        variable b(p);
        minimize (...
            0.5*square_pos(norm(Y-X*b))+...
            delta1*lt'*abs(b)+...
            2*delta2*sum(max(max(abs(b(netwk(:,1)))./wt(netwk(:,1)),abs(b(netwk(:,2)))./wt(netwk(:,2))),abs(b(netwk(:,1)))./wt(netwk(:,1))+abs(b(netwk(:,2)))./wt(netwk(:,2))-tau2))-...
            delta2*(S2(netwk(:,1))'*(b(netwk(:,1))./wt(netwk(:,1))) + S2(netwk(:,2))'*(b(netwk(:,2))./wt(netwk(:,2))))...
            );
    cvx_end
    
    c1=sum(min(abs(b)/tau1,1));
    c2=sum(abs(min((abs(b(netwk(:,1)))./wt(netwk(:,1))) /tau2,1) - min((abs(b(netwk(:,2)))./wt(netwk(:,2)))/tau2,1)));
    s=(0.5*square_pos(norm(Y-X*b))+delta1*tau1*c1+delta2*tau2*c2);
end

