function A = matrixGenerator(n)
% Generate a dense n x n symmetric, positive definite matrix

A = sprandn(n,n,0.1,1e-3); % generate a random n x n matrix

% % construct a symmetric matrix using either
% A = rand(n);
A = 1e-2*eye(n)+A*A';
% % The first is significantly faster: O(n^2) compared to O(n^3)
% 
% % since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
% %   is symmetric positive definite, which can be ensured by adding nI
% A = A + n*eye(n);

% A = gallery('cauchy',1:n);

% A = gallery('chow',n,0.9,1);
% A = gallery('dorr',n,1e-3);
end