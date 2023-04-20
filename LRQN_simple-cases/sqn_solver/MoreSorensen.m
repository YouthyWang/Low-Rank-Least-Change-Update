%% Function: MoreSorensen 
% (Newton's method for trust region sub-problem with safeguarding, proposed
% by More and Sorensen in 1983.)

function vstar = MoreSorensen(A, beta, delta)
% Function MoreSorensen tries to find a numerical solution to trust region subprobelm
% Input
%   A:    matrix of the 2nd order term of trust-region target function
%   beta: vector of the 1st order term of trust-region target function
%   delta:trust-region bound
% Output
%   vstar:  estimated optimal variable v* in the trust region algorithm 
% Ref
%   Computing a Trustregion Step, 1983, by J.J.More & D.C.Sorensen
%
% Main author: Youthy WANG(CUHKSz);
%              youthywyz@gmail.com (oversea); wangyuzh@126.com (China)
%              https://youthywang.github.io/

    lambda = 1;
    m = length(beta);
    Im = eye(m);
    iter = 0;
    maxIter = 1e2;
    lambda1 = eigs(A,1,'smallestabs','Tolerance',1e-10);
    if (abs(lambda1) >= 1e-5)
        A = A - lambda1*Im; % need to add back \lambda_s
    end
    % Initial Values
    lambdas = -min(diag(A));
    lambdal = max([0,lambdas,norm(beta)/delta-norm(A,1)]);
    lambdau = norm(beta)/delta+norm(A,1);
    
    while true
        % Step 1: Safeguarding Scheme
        lambda = max(lambda,lambdal);
        lambda = min(lambda,lambdau);
        if lambda <= lambdas
            lambda = max(1e-3*lambdau, sqrt(lambdal*lambdau));
        end
        [~,f] = chol(A+lambda*Im);
        
        % Step 2: positive definite case
        if (f == 0)
            R = chol(A+lambda*Im);
            q = -(R')^(-1)*beta;
            p = (R)^(-1)*q;
        end
        
        % Step 3: Updating bounds
        if (f==0) && (xi(lambda,A,beta,delta)<0)
            lambdau = min(lambdau,lambda);
        else
            lambdal = max(lambdal, lambda);
        end

        % update lambdas
        if (f==0) && (xi(lambda,A,beta,delta)<0)
            [z,~] = LINPACKtech(R,p,delta);
            lambdas = max(lambdas,lambda-norm(R*z)^2);
        elseif (f > 0)
            [d,u] = UpdateLambdas(A,lambda);
            lambdas = max(lambdas, lambda+d/norm(u)^2);
        end
        lambdal = max(lambdal, lambdas);
        
        % Step 4: check tolerance (convergence)
        if (abs(xi(lambda,A,beta,delta))<1e-5) || (iter>maxIter)
            break;
        end
        % Step 5: updating lambda
        if (f==0) && (norm(beta)~=0)
            q = R'^(-1)*p;
            lambda = lambda + (norm(p)/norm(q))^2*((norm(p)-delta)/delta);
        else
            lambda = lambdas;
        end
        iter = iter + 1;
    end
    vstar = -(A+lambda*Im)^-1*beta;

end

%% Function: LINPACKtech 
% (Use LINPACK techniques to find largest w with U^Tw = e)

function [obj,tau] = LINPACKtech(U,p,delta)
% Function LINPACKtech applies LINPACK technique to find largest w such that U^Tw = e 
% Input
%   U:    Upper triangular matrix from decomposition of original A
%   p:    vector of the 1st order term of trust-region target function
%   delta:trust-region bound
% Output
%   vstar:  estimated optimal variable v* in the trust region algorithm 
% Ref
%   LINPACK User's Guide, 1979.

% NOTE I: SIGN(A,B) = sgn(B)*A (return value of A with sign of B, in C)
% NOTE II: CALL SSCAL(N,SA,SX,INCX): return x = ax (particularly for INCX=1)

    N = length(U);
    A = U'*U;
    ek = 1;
    z = zeros(N,1);
    for k = 1:N
       if z(k) ~= 0
           ek = -z(k)/abs(z(k))*ek;
       end
       if abs(ek-z(k))>abs(A(k,k))
           s = abs(A(k,k))/abs(ek-z(k));
           z = s.*z;
           ek = s*ek;
       end
       wk = ek - z(k);
       wkm = -ek - z(k);
       s = abs(wk);
       sm = abs(wkm);
       if A(k,k) == 0
           wk = 1;
           wkm = 1;
       else
           wk = wk/A(k,k);
           wkm = wkm/A(k,k);
       end
       kp1 = k+1;
       if kp1 <= N
           for j = kp1:N
               sm = sm + abs(z(j)+wkm*A(k,j));
               z(j) = z(j) + wk*A(k,j);
               s = s + abs(z(j));
           end
           if s < sm
               t = wkm - wk;
               wk = wkm;
               for j = kp1:N
                   z(j) = z(j) + t*A(k,j);
               end
           end
       end
       z(k) = wk;
    end
    z = 1/sum(z).*z;
    v = U'^(-1)*z;
    
    obj = v./norm(v);
    tau = (delta^2-norm(p)^2)/(p'*z+((p'*z)/abs(p'*z))*((p'*z)^2+ ...
        (delta^2-norm(p)^2)^(1/2))); 
end

%% Function: UpdateLambdas 
% (A method to update lambdas proposed by Jorege J.More & D.C.Sorensen)

function [d,u] = UpdateLambdas(A,lambda)
    m = length(A);
    if (A(1,1)+lambda <= 0)
        d = -A(1,1)-lambda;
        u = 1;
        return;
    end
    for l = 2:m
        z = -det(A(1:l,1:l)+lambda*eye(l))/det(A(1:l-1,1:l-1)+lambda*eye(l-1));
        if z >= 0
            d = z;
            el = zeros(l,1);
            el(l) = 1;
            NSp = null(A(1:l,1:l)+lambda*eye(l)+d*(el*el'));
            u = NSp(:,1);
            u = u(l)^(-1).*u;
            return;
        end
    end
    d = 0;
    u = 1;
end

function y = xi(t,A,b,s)
    I = eye(length(b));
    y = 1/s-1/norm((A+t*I)^(-1)*b);
end