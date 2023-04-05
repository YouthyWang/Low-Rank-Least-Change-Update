% Codes of algorithms tested: newmethod with quasi-LBFGS update

%% Function: main function newmethodcase 3

function [x1,out] = nmdu3(myFx, x0, gradToler, maxIter, colSize, Q)
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
% Main author: Youthy WANG(CUHKSz);
%              youthywyz@gmail.com (oversea); wangyuzh@126.com (China)
%              https://youthywang.github.io/

% Initialization 
GradH = zeros(1,maxIter);
StepH = zeros(1,maxIter);
FuncH = zeros(1,maxIter);
n = length(x0);
U = zeros(n,colSize);
Im = eye(colSize);
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
    while true
      if k > maxIter
          k = k - 1;
        break;
      end
      % convergence testing
      gnorm = norm(g0);
      done = gnorm < gradToler;
      if done
          fSp = 'Converge (in %i steps)! \n';
          fprintf(fSp,k);
          break;
      end
      % find search direction
      s0 = x1-x0;
      y0 = g1-g0;
      % choose scaling scalar
      low = 0.1; high = 0.9; const = (y0'*s0)/(s0'*s0);
      gamma = min(max(taulst/const-low,0)+low-high, 0)+high;
      tau = gamma*const;
      % prepare the fixed coefficients and vectors
      u_tau = y0-tau*s0;
      vec_Us = U'*s0;
      const_us  = u_tau'*s0;
      const_alpha = ((u_tau'*s0)/(vec_Us'*vec_Us))^(1/2);
      % update low rank matrix U
      if (nnz(U) == 0)
          % trust region degenerates to feasiblity qn
          v = sqrt(sqrt(const_us)/colSize)*ones(colSize,1); 
          U = (u_tau/const_us)*(v');
      else
          U = U + (const_alpha/const_us)*(u_tau-const_alpha*(U*vec_Us))*(vec_Us');
      end 
      % new search direction
      p = (1/tau)*(U*(tau*Im+U'*U)^(-1)*(U'*g1)-g1);
      taulst = tau;

      % line search and update
      [alpha,fs,gs]= strongwolfe(myFx,p,x1,f1,g1,Q);      
      x0 = x1;
      g0 = g1;
      x1 = x1 + alpha*p;
      f1 = fs;
      g1 = gs;
      if (k+1 <= 25 || (mod(k+1,100)==0 && k+1<=1000) || mod(k+1,1000) == 0 || done)
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
end % end of nmdu3