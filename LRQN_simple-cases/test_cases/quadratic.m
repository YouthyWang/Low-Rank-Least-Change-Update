%======================================================
%  Quadratic Function
%  f(x) = (1/2)x^TQx
%  f'(x) = Qx
%======================================================  
function [f,g] = quadratic(x,Q)
%  x : initial point
%  Q : correspoding matrix
%  f : function value
%  g : gradient
%
% example,
%  [f,g] = myfun([1 1 1 1 1 1], eye(6));
%
f = 1/2*transpose(x)*Q*x;
g = zeros(length(x),1);

if nargout > 1
    for i= 1:length(x)
        g(i) = Q(:,i)'*x+Q(i,:)*x;
    end
end

end
