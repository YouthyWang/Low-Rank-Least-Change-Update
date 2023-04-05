% Codes of algorithms tested: widely-used L-BFGS methods

%% Function: main function optLBFGS
function [x1,out] = optLBFGS(myFx, x0, gradToler, maxIter, memSize, Q)
% Function optLBFGS performs multivariate local optimization using the L-BFGS method.
% Input
%   myFx:   the optimized function handle
%   x0:     vector of initial start
%   Q:      mainly for the first test case, no use for others generally 
%   maxIter:max number of iteration  
%   memSize:memory size, 3~30 will be fine
% Output
%   x1:   optimized variable
%   x1:   optimized variable
%   out:  outputs (including history of function values, gradient norms, 
%         and step-sizes) 
%
% Example
%   [optx,out] = optLBFGS(@myfun, x0, 100, 10)
% Notice
%   I don't want to put too many parameters as the inputs, you can set them inside this function.
%   I put all needed functions in this file optLBFGS.m
% Author:
%   Guipeng Li @THU ,  guipenglee@gmail.com

% Initialization
n = length(x0);
Sm = zeros(n,memSize);
Ym = zeros(n,memSize);
GradH = zeros(1,maxIter);
StepH = zeros(1,maxIter);
FuncH = zeros(1,maxIter);
    [f0,g0]=feval(myFx,x0,Q);
    % line search
    % usually line search method only return step size alpha
    % we return 3 variables to save caculation time.
    [alpha,f1,g1] = strongwolfe(myFx,-g0,x0,f0,g0,Q);
    fprintf('%5s %15s %15s %15s\n','iter','step','fval','norm(g)')
    % recordings
    GradH(1) = norm(g1);
    StepH(1) = alpha;
    FuncH(1) = f1;
    x1 = x0 - alpha*g0; % initial estimation of Hessian is set to be identity
    k = 1;
    fprintf('%5d %15.4f %15.4e %15.4e\n',k,alpha,f1,norm(g1));
    while true
      if k > maxIter
          k = k - 1;
        break;
      end

      gnorm = norm(g0);
      if gnorm < gradToler
        fSp = 'Converge (in %i steps)! \n';
        fprintf(fSp,k);
        break;
      end

      % find search direction
      s0 = x1-x0;
      y0 = g1-g0;
      hdiag = s0'*y0/(y0'*y0);
      p = zeros(length(g0),1);
      if (k<=memSize)
        % update S,Y
        Sm(:,k) = s0;
        Ym(:,k) = y0;
        % never forget the minus sign
        p = -getHg_lbfgs(g1,Sm(:,1:k),Ym(:,1:k),hdiag); 
      elseif (k>memSize)
        Sm(:,1:(memSize-1))=Sm(:,2:memSize);
        Ym(:,1:(memSize-1))=Ym(:,2:memSize);
        Sm(:,memSize) = s0;
        Ym(:,memSize) = y0;    
        % never forget the minus sign
        p = -getHg_lbfgs(g1,Sm,Ym,hdiag);
      end  

      % line search
      [alpha,fs,gs]= strongwolfe(myFx,p,x1,f1,g1,Q);
      x0 = x1;
      g0 = g1;
      x1 = x1 + alpha*p;
      f1 = fs;
      g1 = gs;
      % save caculation
      if (k+1 <= 25 || (mod(k+1,100)==0 && k+1<=1000) || mod(k+1,1000) == 0)
        fprintf('%5d %15.4f %15.4e %15.4e\n',k+1,alpha,f1,norm(g1));
      end
      GradH(k+1) = norm(g1);
      StepH(k+1) = alpha;
      FuncH(k+1) = f1;
      k = k+1;    
    end
    out.gh = GradH(1:k);
    out.sh = StepH(1:k);
    out.fh = FuncH(1:k);
    out.it = k;

end% end of optLBFGS

%% Function: getHg_lbfgs (Use L-BFGS method to find the latest search direction)

function Hg = getHg_lbfgs(g,S,Y,hdiag)
% This function returns the approximate inverse Hessian multiplied by the gradient, H*g
% Input
%   S:    Memory matrix (n by k) , s{i}=x{i+1}-x{i}
%   Y:    Memory matrix (n by k) , df{i}=df{i+1}-df{i}
%   g:    gradient (n by 1)
%   hdiag value of initial Hessian diagonal elements (scalar)
% Output
%   Hg    the the approximate inverse Hessian multiplied by the gradient g
% Notice
% This funcion getHg_lbfgs is called by LBFGS_opt.m.
% Ref
%   Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited Storage".
%   Wiki http://en.wikipedia.org/wiki/Limited-memory_BFGS
%   two loop recursion

    [n,k] = size(S);
    for i = 1:k
        ro(i,1) = 1/(Y(:,i)'*S(:,i));
    end

    q = zeros(n,k+1);
    r = zeros(n,1);
    alpha =zeros(k,1);
    beta =zeros(k,1);

    % step 1
    q(:,k+1) = g;

    % first loop
    for i = k:-1:1
        alpha(i) = ro(i)*S(:,i)'*q(:,i+1);
        q(:,i) = q(:,i+1)-alpha(i)*Y(:,i);
    end

    % Multiply by Initial Hessian
    r = hdiag*q(:,1);

    % second loop
    for i = 1:k
        beta(i) = ro(i)*Y(:,i)'*r;
        r = r + S(:,i)*(alpha(i)-beta(i));
    end
     
    Hg=r;
end % end of getHg_lbfgs

