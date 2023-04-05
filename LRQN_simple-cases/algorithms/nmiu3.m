% Codes of algorithms tested: inverse update with quasi-LBFGS update

%% Function: main function newmethodcase 3

function [x1,out] = nmiu3(myFx, x0, gradToler, maxIter, colSize, Q)
% Input
%   myFx:   the optimized function handle
%   x0:     vector of initial start
%   Q:      mainly for the first test case, no use for others generally 
%   maxIter:max number of iteration
%   colSize:the number of columns of updating matrix U
% Output
%   x1:   optimized variable
%   out:  outputs (including history of function values, gradient norms, 
%         and step-sizes)  
%
% Main author: Youthy WANG(CUHKSz);
%              youthywyz@gmail.com (oversea); wangyuzh@126.com (China)
%              https://youthywang.github.io/

% Initialization 
GradH = zeros(1,maxIter);
StepH = zeros(1,maxIter);
FuncH = zeros(1,maxIter);
n = length(x0);
U = zeros(n,colSize);
S = zeros(n,colSize/2);
Y = zeros(n,colSize/2);
taulst = 0;

    [f0,g0]=feval(myFx,x0,Q);
    % line search
    % usually line search method only return step size alpha
    % we return 3 variables to save caculation time.
    [alpha,f1,g1] = strongwolfe(myFx,-g0,x0,f0,g0,Q);
    fprintf('%5s %15s %15s %15s\n','iter','step','fval','norm(g)');
    % recordings
    GradH(1) = norm(g1);
    StepH(1) = alpha;
    FuncH(1) = f1;
    x1 = x0 - alpha*g0; 
    k = 1;
    fprintf('%5d %15.4f %15.4e %15.4e\n',k,alpha,f1,norm(g1));
   % loop for optimization algorithms
    while true
      % stop: maximum iteration
      if k > maxIter
          k = k - 1;
        break;
      end
      % convergence testing
      gnorm = norm(g0);
      done = gnorm < gradToler;
      % stop: small enough gradient norm
      if done
          fSp = 'Converge (in %i steps)! \n';
          fprintf(fSp,k);
          break;
      end
      % find search direction
      s0 = x1-x0;
      y0 = g1-g0;
      % BFGS for initial steps (Codes from ``Guipeng Li @THU'')
      if (k <= (colSize/2)) 
          % update S,Y
          hdiag = (s0'*y0)/(y0'*y0);
          S(:,k) = s0;
          Y(:,k) = y0;
          % never forget the minus sign
          p = -getHg_lbfgs(g1,S(:,1:k),Y(:,1:k),hdiag); 
          % construct initial low-rank matrix U
          if (k == (colSize/2))
              Dk = diag(diag(S'*Y));
              Rk = triu(S'*Y);
              Rkinv = Rk^(-1);
              [V,D] = eig([Rkinv'*(Dk+hdiag.*(Y'*Y))*Rkinv,-Rkinv';-Rkinv,zeros(k)]);
              D = max(D,0); % drop all negative eigenvalues
              U = [S,hdiag.*Y]*(V*(D)^(1/2)*V');
          end
      else
          % choose scaling scalar
          low = 0.1; high = 0.9; const_yts = y0'*s0; const_yty = y0'*y0;
          const = const_yts/const_yty;
          gamma = min(max(taulst/const-low,0)+low-high, 0)+high;
          tau = gamma*const;
          % prepare the fixed coefficients and vectors
          u_tau = s0-tau*y0;
          const_uy  = const_yts-tau*const_yty;
          vec_Uty = U'*y0;
          const_alpha = (const_uy/(vec_Uty'*vec_Uty))^(1/2);
          % update low rank matrix U
          U = U + (const_alpha/const_uy)*(u_tau-const_alpha*U*vec_Uty)*vec_Uty';
          p = -tau*g1-U*(U'*g1);% new search direction
          taulst = tau;
      end
      % line search
      [alpha,fs,gs]= strongwolfe(myFx,p,x1,f1,g1,Q);      
      x0 = x1;
      g0 = g1;
      x1 = x1 + alpha*p;
      f1 = fs;
      g1 = gs;
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
end % end of nmiu3

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