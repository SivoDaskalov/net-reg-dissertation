%% Methods
%  Lasso, Grace, Enet, aGrace, Linf_w, aLinf_w, TLP+TLPI_w, Lasso+TLPI_w
%  Through the program, I use sqrt(d) as weight
function[MEPE,TPFP,bsdmse,b_lasso,b_Grace,b_Enet,b_aGrace,b_Linf,b_aLinf,b_TLP,b_LTLP]=TIP(b,Tr,Tu,Te,Rep,rho,netwk)

nnetwk=10;

epsilon=1e-6;

%%data
 format shortG
 [tn,p]=size(Tr(:,2:end));
 n=tn/Rep; 
 
 tm=size(Te(:,2:end),1);
 m=tm/Rep; 
 
%wt,w=sqrt(d)
a=netwk(:); 
wt=ones(p,1);
for j=1:p
 %weight=1
 %wt(j)= 1;
 %weight=sqrt(d)
  wt(j)= sqrt(sum(a==j));
end

netp=p/nnetwk;
 
%TP,FP locations
   q1= abs(b)>0;  
   q0= abs(b)==0; 
   
%% covX=E[X'X] 
 covX1=ones(netp,netp)*n*rho^2;
 covX1(1:netp+1:end)=n;
  a=cell(1,nnetwk);
 [a{:}]=deal(sparse(covX1));
 samcovX=blkdiag(a{:});
 covX=full(samcovX);
 
%% Center testing data
tX=Tr(:,2:end);
tY=Tr(:,1);

tXtu=Tu(:,2:end);
tYtu=Tu(:,1);

tXte=Te(:,2:end);
tYte=Te(:,1);


%% Summary Table
ME=zeros(Rep,8);PE=zeros(Rep,8);
TP=zeros(Rep,8);FP=zeros(Rep,8);

b_lasso=zeros(Rep,p);
b_Grace=zeros(Rep,p);
b_Enet=zeros(Rep,p);
b_aGrace=zeros(Rep,p);
b_Linf=zeros(Rep,p);
b_aLinf=zeros(Rep,p);
b_TLP=zeros(Rep,p);
b_LTLP=zeros(Rep,p);
%b_ncTFGS=zeros(Rep,p);

bs_lasso=zeros(Rep,p);
bs_Grace=zeros(Rep,p);
bs_Enet=zeros(Rep,p);
bs_aGrace=zeros(Rep,p);
bs_Linf=zeros(Rep,p);
bs_aLinf=zeros(Rep,p);
bs_TLP=zeros(Rep,p);
bs_LTLP=zeros(Rep,p);
%bs_ncTFGS=zeros(Rep,p);

%% Repeat 
for d =1:Rep
 
 disp(d)
%% (Y,X) , (Ytu,Xtu), (Yte,Xte). all sample size are same as n. o.w change!
 Y=tY((1+n*(d-1)):(n*d),1); 
 X=tX((1+n*(d-1)):(n*d),:);  
 Ytu=tYtu((1+n*(d-1)):(n*d),1);
 Xtu=tXtu((1+n*(d-1)):(n*d),:);
 Yte=tYte((1+m*(d-1)):(m*d),1);
 Xte=tXte((1+m*(d-1)):(m*d),:);
%% Lasso
  lam=100;
  bls=lasso(Y,X,lam); 
  cv=[]; lamlasso=[];
   while (sum(abs(bls)>0.0001)>0)
     bls=lasso(Y,X,lam); 
     cv=[cv norm(Ytu-Xtu*bls)^2];
     lamlasso=[lamlasso lam];
     lam=lam-1;
   end
   ind= cv==min(cv);
   blasso=lasso(Y,X,lamlasso(ind));    

%% Grace
a1=0.1; b1=1000; a2=0.1; b2=1000;
K1=10; K2=10;
r1 = exp(log(b1/a1)/K1);
r2 = exp(log(b2/a2)/K2);

a=ones(size(netwk,1),1); %only difference with aGrace here

minPSE1 = 1e+10;
x1 = zeros(p, 1);
lam11 = 0;
lam21 = 0;
lam1 = a1/r1;
for i=0:K1,
	lam1 = lam1*r1;
	lam2 = a2/r2;
	for j=0:K2,
		lam2 = lam2*r2;
		[x,flag] = cvxLiListep2(Y, X, wt, netwk, a, lam1, lam2);
		PSEtu = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
		if(PSEtu < minPSE1)
				minPSE1 = PSEtu;
				x1 = x;
				lam11 = lam1;
				lam21 = lam2;
		end
	end
end
bGrace=x1;

%% Enet
a1=0.1; b1=1000; a2=0.1; b2=1000;
K1=10; K2=10;
r1 = exp(log(b1/a1)/K1);
r2 = exp(log(b2/a2)/K2);

lam10 = 0;
lam20 = 0;
x0 = zeros(p, 1);
minPSE0 = 1e+10;

	lam1 = a1/r1;
	for i=0:K1,
		lam1 = lam1*r1;
	  lam2 = a2/r2;
		for j=0:K2,
			lam2 = lam2*r2;
			[x,flag] = cvxEnet(Y, X, lam1, lam2);
			PSEtu = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
			if(PSEtu < minPSE0)
					minPSE0 = PSEtu;
					x0 = x;
					lam10 = lam1;
					lam20 = lam2;
			end
	  end
  end
bEnet=x0;

%% aGrace 
   if(p<n)
	 [x, flag] = cvxLS(Y, X);
	 minPSE0 = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
	 x0 = x;
   else
     x0=bEnet;
   end    
   % extract information from the old solution
x = x0;
a=(x(netwk(:,1)).*x(netwk(:,2))>epsilon*epsilon).*(abs(x(netwk(:,1)))>epsilon).*(abs(x(netwk(:,2)))>epsilon); % a=1 if x_i*x_j>0; a=0 otherwise
a=2*a-1; % a=1 if x_i*x_j>0; a=-1 otherwise

% fit the new model
minPSE1 = 1e+10;
x1 = zeros(p, 1);
lam11 = 0;
lam21 = 0;
lam1 = a1/r1;
for i=0:K1,
	lam1 = lam1*r1;
	lam2 = a2/r2;
	for j=0:K2,
		lam2 = lam2*r2;
		[x,flag] = cvxLiListep2(Y, X, wt, netwk, a, lam1, lam2);
		PSEtu = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
		if(PSEtu < minPSE1)
				minPSE1 = PSEtu;
				x1 = x;
				lam11 = lam1;
				lam21 = lam2;
		end
	end
end
baGrace=x1;

%% Linf_w 
%fit the original model
gamma=Inf;
[x,flag] = cvxquadprog(Y,X,netwk,wt,gamma,100); % the solution with virtually no constraint 
maxC = ceil(netwknorm(x,netwk,wt,gamma));
%C = 0.5:0.5:maxC;
C = 1:1:maxC;
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
bLinf=x0;
%% aLinf_w
% extract information from the old solution
x = x0;
tmp=(abs(x(netwk(:,1)))>epsilon)|(abs(x(netwk(:,2)))>epsilon);
netwk2=netwk(tmp==1,:); % edges with at least one informative node, not used
dnetwk=netwk(tmp==0,:); % edges whose both nodes are non-informative
dm = reshape(dnetwk',[], 1);
dm = unique(dm); % which x_i to be removed

a=(x(netwk(:,1)).*x(netwk(:,2))>epsilon*epsilon).*(abs(x(netwk(:,1)))>epsilon).*(abs(x(netwk(:,2)))>epsilon); % a=1 if x_i*x_j>0; a=0 otherwise
a=2*a-1; % a=1 if x_i*x_j>0; a=-1 otherwise

% fit the new model
     C=0:0.3:30;
%C = 1:1:maxC;

k = size(C,2);
[x,flag] = cvxnetprog0(Y,X,netwk,wt,a,dm); % the solution with C=0
minPSE1 = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
x1 = x;
C1 = C(1);
for i=2:k,
    [x,flag] = cvxnetprog(Y,X,netwk,wt,a,dm,C(i));
    PSEtu = dot(Ytu-Xtu*x,Ytu-Xtu*x)/length(Ytu);
    if(PSEtu < minPSE1)
        minPSE1 = PSEtu;
        x1 = x;
        C1 = C(i);
    end
end
bLinf2=x1;
%% TLPI+TLPI_w
gd1=4;gd2=4;gd3=5;
%Constraint version
%      ag=max(abs(blasso));
%      K1=linspace(p/3,p,gd1);
%      nedge=size(netwk,1);
%      K2=linspace(nedge,nedge*1.5,gd2);
%      ltau=linspace(0.01,ag/2,gd3); 
% Lagrange    
      mx=max(abs(blasso));
      K1=linspace(mx,mx/4*p,gd1); %del1
      edge=size(netwk,1);
      K2=linspace(mx,mx*edge,gd2);  %del2
      ltau=linspace(epsilon,mx/2,gd3); 

   tunM=zeros(gd1,gd2*gd3);
    for j=1:gd1
       for k=1:gd2
         for s=1:gd3
                btr=cvxtip(Y,X,K1(j),K2(k),ltau(s),ltau(s),blasso,netwk,wt);  
                tunM(j,(k-1)*gd3+s)=norm(Ytu-Xtu*btr)^2;        
          end
       end
    end
       tmp=tunM>0; 
       [row,col]=find(tunM==min(tunM(tmp)));
       row=row(1);col=col(1);
       del1=mean(K1(row));
        if   rem(col,gd3)==0, del2=mean(K2(floor(col/gd3)));   tau=mean(ltau(gd3));
          else del2=mean(K2(floor(col/gd3)+1)); tau=mean(ltau(col-floor(col/gd3)*gd3));      
        end
     bTLP=cvxtip(Y,X,del1,del2,tau,tau,blasso,netwk,wt);

%% Lasso+TLPI_w w/ Lagrange form
   LtunM=zeros(gd1,gd2*gd3);
   
   K1=linspace(lamlasso(ind)/1.5,lamlasso(ind)*1.5,gd1);
   
    for j=1:gd1
       for k=1:gd2
         for s=1:gd3
                btr=cvxtip(Y,X,K1(j),K2(k),100,ltau(s),blasso,netwk,wt);  
                LtunM(j,(k-1)*gd3+s)=norm(Ytu-Xtu*btr)^2;        
          end
       end
    end
       tmp=LtunM>0; 
       [row,col]=find(LtunM==min(LtunM(tmp)));
       row=row(end);col=col(end);
       del1=mean(K1(row));
        if   rem(col,gd3)==0, del2=mean(K2(floor(col/gd3)));   tau=mean(ltau(gd3));
          else del2=mean(K2(floor(col/gd3)+1)); tau=mean(ltau(col-floor(col/gd3)*gd3));      
        end
     bLTLP=cvxtip(Y,X,del1,del2,100,tau,blasso,netwk,wt);

%% add TTLP from Sen Yang
%  mx=max(abs(blasso));
%  gd=5;
%  lamf=linspace(0.001,5,gd); 
%  lams=linspace(0.001,5,gd); 
%  ltau=linspace(0.001,mx*0.5,gd); 
% 
% tun_ncTFGS=zeros(gd,gd*gd); 
% opts=[];
%   for s=1:gd
%      %disp(s)
%     for t=1:gd
%       for u=1:gd
%       opts.tau=ltau(u);
%       x0 = ncTFGS(X,Y,netwk',lamf(s),lams(t),opts);
%       tun_ncTFGS(s,(t-1)*gd+u)=norm(Ytu-Xtu*x0)^2;   
%       end  
%    end
%   end   
% 
%        tmp=tun_ncTFGS>0; 
%        [row,col]=find(tun_ncTFGS==min(tun_ncTFGS(tmp)));
%        row=row(1);col=col(1);
%        lam1=mean(lamf(row));
%         if   rem(col,gd)==0, lam2=mean(lams(floor(col/gd)));   tau=mean(ltau(gd));
%           else lam2=mean(lams(floor(col/gd)+1)); tau=mean(ltau(col-floor(col/gd)*gd));      
%         end
% 
%   opts.tau=tau;        
%   x1 = ncTFGS(X,Y,netwk',lam1,lam2,opts);
%   bncTFGS=x1;   
%% Result
 %bhat
   b_lasso(d,:)=blasso;
   b_Grace(d,:)=bGrace;
   b_Enet(d,:)=bEnet;
   b_aGrace(d,:)=baGrace;
   b_Linf(d,:)=bLinf;
   b_aLinf(d,:)=bLinf2;
   b_TLP(d,:)=bTLP;
   b_LTLP(d,:)=bLTLP;
   %b_ncTFGS(d,:)=bncTFGS;
   
 %bias^2
 bs_lasso(d,:)=(blasso-b).*(blasso-b);
 bs_Grace(d,:)=(bGrace-b).*(bGrace-b);
 bs_Enet(d,:)=(bEnet-b).*(bEnet-b);
 bs_aGrace(d,:)=(baGrace-b).*(baGrace-b);
 bs_Linf(d,:)=(bLinf-b).*(bLinf-b);
 bs_aLinf(d,:)=(bLinf2-b).*(bLinf2-b);
 bs_TLP(d,:)=(bTLP-b).*(bTLP-b);
 bs_LTLP(d,:)=(bLTLP-b).*(bLTLP-b);
 %bs_ncTFGS(d,:)=(bncTFGS-b).*(bncTFGS-b);
 
 %ME
   ME(d,1)=(b-blasso)'*covX*(b-blasso)/length(Y);
   ME(d,2)=(b-bGrace)'*covX*(b-bGrace)/length(Y);
   ME(d,3)=(b-bEnet)'*covX*(b-bEnet)/length(Y);
   ME(d,4)=(b-baGrace)'*covX*(b-baGrace)/length(Y);
   ME(d,5)=(b-bLinf)'*covX*(b-bLinf)/length(Y);
   ME(d,6)=(b-bLinf2)'*covX*(b-bLinf2)/length(Y);
   ME(d,7)=(b-bTLP)'*covX*(b-bTLP)/length(Y);
   ME(d,8)=(b-bLTLP)'*covX*(b-bLTLP)/length(Y);
   %ME(d,9)=(b-bncTFGS)'*covX*(b-bncTFGS)/length(Y);
 %PE
   PE(d,1)=dot(Yte-Xte*blasso,Yte-Xte*blasso)/length(Yte);
   PE(d,2)=dot(Yte-Xte*bGrace,Yte-Xte*bGrace)/length(Yte);
   PE(d,3)=dot(Yte-Xte*bEnet,Yte-Xte*bEnet)/length(Yte);
   PE(d,4)=dot(Yte-Xte*baGrace,Yte-Xte*baGrace)/length(Yte);
   PE(d,5)=dot(Yte-Xte*bLinf,Yte-Xte*bLinf)/length(Yte);
   PE(d,6)=dot(Yte-Xte*bLinf2,Yte-Xte*bLinf2)/length(Yte);
   PE(d,7)=dot(Yte-Xte*bTLP,Yte-Xte*bTLP)/length(Yte);
   PE(d,8)=dot(Yte-Xte*bLTLP,Yte-Xte*bLTLP)/length(Yte);
   %PE(d,9)=dot(Yte-Xte*bncTFGS,Yte-Xte*bncTFGS)/length(Yte);
 %TP
   TP(d,1) = sum(abs(blasso(q1))>0.001); 
   TP(d,2) = sum(abs(bGrace(q1))>0.001);
   TP(d,3) = sum(abs(bEnet(q1))>0.001);
   TP(d,4) = sum(abs(baGrace(q1))>0.001);
   TP(d,5) = sum(abs(bLinf(q1))>0.001);
   TP(d,6) = sum(abs(bLinf2(q1))>0.001);
   TP(d,7) = sum(abs(bTLP(q1))>0.001);
   TP(d,8) = sum(abs(bLTLP(q1))>0.001);
   %TP(d,9) = sum(abs(bncTFGS(q1))>0.001);
 %FP
   FP(d,1) = sum(abs(blasso(q0))>0.001); 
   FP(d,2) = sum(abs(bGrace(q0))>0.001);
   FP(d,3) = sum(abs(bEnet(q0))>0.001);
   FP(d,4) = sum(abs(baGrace(q0))>0.001);
   FP(d,5) = sum(abs(bLinf(q0))>0.001);
   FP(d,6) = sum(abs(bLinf2(q0))>0.001);
   FP(d,7) = sum(abs(bTLP(q0))>0.001);
   FP(d,8) = sum(abs(bLTLP(q0))>0.001);
   %FP(d,9) = sum(abs(bncTFGS(q0))>0.001);

  %interim check
  disp([mean(ME(1:d,:)); mean(PE(1:d,:)); mean(TP(1:d,:)); mean(FP(1:d,:))])
end

%% Summary
MEPE=[mean(ME)' sqrt(var(ME))' mean(PE)' sqrt(var(PE))'];
TPFP=[mean(TP)' sqrt(var(TP))' mean(FP)' sqrt(var(FP))'];
bsdmse=[mean(b_lasso)' sqrt(var(b_lasso))'  (mean(bs_lasso)+var(b_lasso))' mean(b_Grace)'  sqrt(var(b_Grace))' (mean(bs_Grace)+var(b_Grace))' mean(b_Enet)'   sqrt(var(b_Enet))'   (mean(bs_Enet)+var(b_Enet))' mean(b_aGrace)' sqrt(var(b_aGrace))' (mean(bs_aGrace)+var(b_aGrace))' mean(b_Linf)'   sqrt(var(b_Linf))'   (mean(bs_Linf)+var(b_Linf))' mean(b_aLinf)'  sqrt(var(b_aLinf))'  (mean(bs_aLinf)+var(b_aLinf))' mean(b_TLP)'  sqrt(var(b_TLP))'  (mean(bs_TLP)+var(b_TLP))' mean(b_LTLP)' sqrt(var(b_LTLP))' (mean(bs_LTLP)+var(b_LTLP))'];

