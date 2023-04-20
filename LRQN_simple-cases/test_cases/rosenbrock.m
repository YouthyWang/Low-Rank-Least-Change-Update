
function [f, df, ddf] = rosenbrock(x,Q)

% rosenbrock.m 
% This function returns the function value, partial derivatives
% and Hessian of the (general dimension) rosenbrock function, given by:
%
%       f(x) = sum_{i=1:D-1} Q*(x(i+1) - x(i)^2)^2 + (1-x(i))^2 
%
% where D is the dimension of x. The true minimum is 0 at x = (1 1 ... 1).
%
% Carl Edward Rasmussen, 2001-07-21.

D = length(x);
f = sum(Q*(x(2:D)-x(1:D-1).^2).^2 + (1-x(1:D-1)).^2);

if nargout > 1
  df = zeros(D, 1);
  df(1:D-1) = - 4*Q*x(1:D-1).*(x(2:D)-x(1:D-1).^2) - 2*(1-x(1:D-1));
  df(2:D) = df(2:D) + 2*Q*(x(2:D)-x(1:D-1).^2);
end

if nargout > 2
  ddf = zeros(D,D);
  ddf(1:D-1,1:D-1) = diag(-4*Q*x(2:D) + 12*Q*x(1:D-1).^2 + 2);
  ddf(2:D,2:D) = ddf(2:D,2:D) + 2*Q*eye(D-1);
  ddf = ddf - diag(4*Q*x(1:D-1),1) - diag(4*Q*x(1:D-1),-1);
end
