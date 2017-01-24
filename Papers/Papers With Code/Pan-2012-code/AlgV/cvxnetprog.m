%% function name: cvxnetprog
% solves the following problem:
% min |Ax-b|
% s.t. \sum_{i~j} |x(i)/w(i)-a(i,j)*x(j)/w(j)| <= C
%     x(d(:)) == 0;
% where a(i,j) = 1 if x_old(i)*x_old(j)>=0, and -1 otherwise
function [y,exitflag] = cvxnetprog(b, A, netwk, wt, a, d, C)
p = size(A,2);
cvx_begin quiet
  cvx_solver sdpt3
  variable x(p);
  minimize(norm(b-A*x));
  subject to
    sum(abs(x(netwk(:,1))./wt(netwk(:,1))-a(:).*x(netwk(:,2))./wt(netwk(:,2)))) <= C;
    x(d(:)) == 0;
cvx_end
exitflag = cvx_status;
y=x;
