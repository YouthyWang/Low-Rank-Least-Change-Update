% test script for unconstrained optimization
clear, close all

%% choose examples to test
% example descriptions
Examples = {'quadratic','rank1approx','rosenbrock'};
FuncDescrp = {'quadratic function', ...
    'rank 1 approximation', ...
    'ronsenbrock function'};
formatSpec = 'Enter %i to test example %5s (<strong> %5s </strong>)\r\n';
for i = 1:length(Examples)
    fprintf(formatSpec, i, Examples{i}, FuncDescrp{i});
end
% choose codes user interface
explChosen = input('Choose an example to test (default 1): ');

%% problem size and initial data
if (explChosen == 2)
    size = input('Problem size (default 100): ');
    if isempty(size), size = 100; end
    x0 = ones(size,1);
    Q = matrixGenerator(size);
    en = 2;
elseif (explChosen == 3) 
    size = input('Problem size (default 10): ');
    if isempty(size), size = 10; end
    x0 = zeros(size,1);
    Q = 100; % useless
    en = 3;
else
    size = input('Problem size (default 100): ');
    if isempty(size), size = 100; end
    x0 = ones(size,1);
    Q = matrixGenerator(size);
    en = 1;
end

%% initialization
Codes = {'optLBFGS','nmdu1','nmdu3','nmiu1','nmiu2','nmiu3'}; % function names
maxIter = 1e3; % default maxmium iteration number
Time = zeros(6,1);
tol = 1e-7;
Objv = ones(6,1)*Inf; 
It = zeros(6,1);
Gh = cell(6,1);
Sh = cell(6,1);
Fh = cell(6,1);
para1 = 3; % memory size for L_BFGS, and no use for others
para2 = 6; % column size for new methods, and no use for others

%% run solvers

for j = 1:6

    solver = Codes{j}; 
    fprintf(['\n--- Run ' solver ' ---\n'])
    t0 = tic;
    if (j==1)
        [x,out] = eval([solver '(@' Examples{en} ',x0,tol,maxIter,para1,Q);']);
    else
        [x,out] = eval([solver '(@' Examples{en} ',x0,tol,maxIter,para2,Q);']);
    end
    toc(t0)
    Time(j) = toc(t0);
    [Objv(j),~] = eval([Examples{en} '(x,Q)']);
    It(j) = out.it;
    Gh{j} = out.gh;
    Sh{j} = out.sh;
    Fh{j} = out.fh;
    
end

fprintf('\nSolvers:\n'), fprintf('\t%s\n',Codes{1:6})
fprintf('\nObj value:\n'); format long;  disp(Objv)
fprintf('\nTime used:\n'); format short; disp(Time)

figure(1)
h1 = semilogy(1:It(1),Gh{1},1:It(2),Gh{2},1:It(3),Gh{3},1:It(4),Gh{4},...
    1:It(5),Gh{5},1:It(6),Gh{6});
legend(Codes{1},Codes{2},Codes{3},Codes{4},Codes{5},Codes{6});

set(h1,'linewidth',2); grid on
title('Gradient-Norm History')
xlabel('Iterations')

figure(2)
h2 = semilogy(1:It(1),Fh{1},1:It(2),Fh{2},1:It(3),Fh{3},1:It(4),Fh{4},...
    1:It(5),Fh{5},1:It(6),Fh{6});
legend(Codes{1},Codes{2},Codes{3},Codes{4},Codes{5},Codes{6});

set(h2,'linewidth',2); grid on
title('Function Value History')
xlabel('Iterations')
