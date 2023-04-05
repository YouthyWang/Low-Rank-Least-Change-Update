function v = Ye(A, b, bdd)
    % Step I: Scaling and Shifting
    c = -b./bdd; % need to transform back: v = bdd*x
    lambdas = eigs(A,1,'smallestabs','Tolerance',1e-10);
    if (abs(lambdas) < 1e-5)
        Q = A;
    else
        Q = A - lambdas*eye(length(A)); % need to add back \lambda_s
    end
    x_test = pinv(Q)*c;
    % choose of epsilon (conditions in the paper)
    p = poly(Q);
    p_nz = p(abs(p)>1e-5);
    eps = min(0.5, abs(p_nz(end))/(max([abs(p_nz),1])+abs(p_nz(end)))); % case II
    if (norm(Q*x_test-c) >= 1e-5*norm(c))
        eps = min(eps, norm(null(Q)'*c)); % case I
    end
    % Step II: Check whether there is a solution for Qx=c
    if (norm(Q*x_test-c) < 1e-5*norm(c)) && (norm(x_test) < 1+eps)
        % find v in null space of Q s.t. pinv(Q)c+v is close to 1
        R = chol(Q+1e-5*eye(length(Q)));
        [z,tau] = LINPACKtech(R,x_test,bdd);
        if (~isreal(tau))
            tau = abs(tau);
        end
        v = bdd.*(x_test+tau.*z);
        return;
    % Step III: Bisection and Newton search
    else
        upp_bd = 1;
        % correction of numerical errors
        while (xi(eps^3,Q,c,1) < 0)
            eps = 0.9*eps;
        end
        low_bd = eps^3;
        beta = 1 + 1/12;
        K = abs(ceil(log2(log2(1/low_bd))-log2(log2(beta))));
        l_bin = beta^(2^(K-1))*low_bd;
        for k = K-1:-1:1 % binary search in K steps reduces the interval length to target
            if (xi(l_bin,Q,c,1) > 0)
                low_bd = l_bin;
                l_bin = beta^(2^(k-1))*l_bin;
            elseif (xi(l_bin,Q,c,1) < 0)
                upp_bd = l_bin;
                l_bin = beta^(-2^(k-1))*l_bin;
            else 
                lambda = l_bin;
                v = bdd.*(Q+lambda*eye(length(Q))^(-1)*c);
                return;
            end
        end
        % Do Newton search in the small interval
        lambda = low_bd;

        while (abs(xi(lambda,Q,c,1)) > 1e-5)
            R = chol(Q+lambda*eye(length(Q)));
            p = (R'*R)^-1*c;
            q = R'^(-1)*p;
            lambda = lambda + (norm(p)/norm(q))^2*((norm(p)-1));
        end
        v = bdd.*(pinv(Q+lambda*eye(length(Q)))*c);
    end
end

function y = xi(t,A,b,s)
    I = eye(length(b));
    y = 1/s-1/norm((A+t*I)^(-1)*b);
end
