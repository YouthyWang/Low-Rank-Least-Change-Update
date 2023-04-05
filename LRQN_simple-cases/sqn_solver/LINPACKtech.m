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