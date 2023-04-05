%======================================================
%  Rank 1 Approximation of A Matrix
%  f(x) = (1/4)||xx^T-Q||^2 = (1/4)*[(x^Tx)^2-2x^TQx]
%  f'(x) = [(x^Tx)I-Q]x
%======================================================  

function [f, df] = rank1approx(x,Q)
%  x : initial point
%  Q : correspoding matrix
%  f : function value
%  g : gradient
%
% example,
%  [f,g] = myfun([1 1 1 1 1 1], eye(6));;
%

f = 0.25*norm(x*x'-Q,'fro')^2;
df = (x'*x).*x-Q*x;

end
