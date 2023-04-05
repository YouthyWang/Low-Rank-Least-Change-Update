% Codes of algorithms tested: newmethod with I as weighted matrix

%% Function: main function newmethodcase 1

function [x1,out] = nmdu1(myFx, x0, gradToler, maxIter, colSize, Q)
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
S = zeros(n,colSize);
Y = zeros(n,colSize);
H0 = eye(n);
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
      % find new search direction
      s0 = x1-x0;
      y0 = g1-g0;
      % SR1 for initial steps
      if (k <= colSize) 
          S(:,k) = s0; 
          Y(:,k) = y0;
          H0 = H0+(s0-H0*y0)/((s0-H0*y0)'*y0)*(s0-H0*y0)';
          p = -H0*g1; % new search direction
          % construct initial low-rank matrix U
          if (k == colSize)
              Dk = diag(diag(S'*Y));
              Lk = tril(S'*Y)-Dk;             
              [V,D] = eig((Dk+Lk+Lk'-S'*S)^(-1));
              D = max(D,0); % drop all negative eigenvalues
              U = (Y-S)*(V*(D)^(1/2)*V');
          end
      else
          % choose scaling scalar
          low = 0.1; high = 0.9; const = (y0'*s0)/(s0'*s0); 
          gamma = min(max(taulst/const-low,0)+low-high, 0)+high; 
          tau = gamma*const; 
          % prepare the fixed coefficients and vectors
          u_tau = y0-tau*s0;
          const_ss = s0'*s0;
          const_us  = u_tau'*s0;
          vec_Us = (U'*s0);
          % sub-pquestion: quadratic minimization spherical constraint
          A = U'*U-(1/(const_ss))*(vec_Us*vec_Us'); % disp(det(A)),
          if (nnz(U) == 0)
              % trust region degenerates to feasiblity qn
              vstar = sqrt(sqrt(const_us)/colSize)*ones(colSize,1); 
          else
              beta = -U'*u_tau; 
              vstar = MoreSorensen(A,beta,sqrt(const_us));  % trust region algorithm to find v*
          end 
          % prepare the fixed coefficients and vectors
          vec_Uv = U*vstar;
          % update low-rank matrix U
          U = U + ((s0'*vec_Uv)/(const_us*const_ss)*s0+(1/const_us)*...
              (u_tau-vec_Uv))*vstar' - 1/const_ss*s0*(vec_Us'); 
          p = (1/tau)*(U*(tau*Im+U'*U)^(-1)*(U'*g1)-g1);% new search direction
          taulst = tau;
      end

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
end % end of nmdu1
