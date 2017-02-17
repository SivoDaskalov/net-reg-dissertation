epsilon=1e-6;%cut-off value for non-informative genes

if(small==1)
	nInform=22;
else
	nInform=44;
end

%% read in the data
X = dlmread('Xtr.txt', ' ');
Y = dlmread('Ytr.txt', ' ');
Xtu = dlmread('Xtu.txt', ' ');
Ytu = dlmread('Ytu.txt', ' ');
Xts = dlmread('Xts.txt', ' ');
Yts = dlmread('Yts.txt', ' ');
b = dlmread('b.txt', ' ');
netwk = dlmread('networkConf.txt', ' ');
wt = dlmread('wt.txt',' ');
p = size(X,2);

%% fit the original model
[x,flag] = cvxquadprog(Y,X,netwk,wt,gamma,1000); % the solution with virtually no constraint 
maxC = ceil(netwknorm(x,netwk,wt,gamma));
C = 0.5:0.5:maxC;
k = size(C,2);

[x,flag] = cvxquadprog(Y, X, netwk, wt, gamma, C(1));
minPSE0 = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
x0 = x;
C0 = C(1);
for i=2:k,
  [x,flag] = cvxquadprog(Y, X, netwk, wt, gamma, C(i));
  PSEtu = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
  if(PSEtu < minPSE0)
      minPSE0 = PSEtu;
      x0 = x;
      C0 = C(i);
  end
end

%% extract information from the old solution
x = x0;
tmp=(abs(x(netwk(:,1)))>epsilon)|(abs(x(netwk(:,2)))>epsilon);
netwk2=netwk(tmp==1,:); % edges with at least one informative node
dnetwk=netwk(tmp==0,:); % edges whose both nodes are non-informative
d = reshape(dnetwk',[], 1);
d = unique(d); % which x_i to be removed
a=(x(netwk(:,1)).*x(netwk(:,2))>epsilon*epsilon).*(abs(x(netwk(:,1)))>epsilon).*(abs(x(netwk(:,2)))>epsilon); % a=1 if x_i*x_j>0; a=0 otherwise
a=2*a-1; % a=1 if x_i*x_j>0; a=-1 otherwise

%% fit the new model
if (small==1)
	C=0:0.5:30;
else
    C=0:0.3:30;
end
k = size(C,2);
[x,flag] = cvxnetprog0(Y,X,netwk,wt,a,d); % the solution with C=0
minPSE1 = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
x1 = x;
C1 = C(1);
for i=2:k,
    [x,flag] = cvxnetprog(Y,X,netwk,wt,a,d,C(i));
    PSEtu = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
    if(PSEtu < minPSE1)
        minPSE1 = PSEtu;
        x1 = x;
        C1 = C(i);
    end
end

%% prediction performance
oldPSEts=dot(Yts-Xts*x0,Yts-Xts*x0)/length(Yts);
oldq1 = sum(abs(x0(1:nInform))<epsilon);
oldq0 = sum(abs(x0((nInform+1):p))<epsilon);

PSEts=dot(Yts-Xts*x1,Yts-Xts*x1)/length(Yts);
q1 = sum(abs(x1(1:nInform))<epsilon);
q0 = sum(abs(x1((nInform+1):p))<epsilon);

%% save results to file
fid=fopen(outfilename,'a');
fprintf(fid, '%6.2f %6.2f %6.2f %d %d', C0, oldPSEts, minPSE0, oldq1, oldq0);
for i=1:p,
    fprintf(fid, ' %6.2f', x0(i));
end
fprintf(fid, ' %6.2f %6.2f %6.2f %d %d', C1, PSEts, minPSE1, q1, q0);
for i=1:p,
    fprintf(fid, ' %6.2f', x1(i));
end
fprintf(fid, '\n');
fclose(fid);

exit;
