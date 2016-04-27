function [alphas, beta0] = SMO(K, y, C)
%SMO: Implementation of Sequential Minimal Optimization

% -------------------------------------
% INITIALIZATION
% -------------------------------------

% Verbose option
verbose = 1;

% Tolerance by which the KKT conditions and stopping criteria are checked
tolKKT = 10^-6;

% Maximum number of iterations
maxIterations = 10000;

% Tolerance by which support vectors are identified (compare svmtrain function):
svTol = sqrt(eps);

% Number of data points:
nPoints = length(y);

% Constraints for each data point 
boxConstraints = ones(nPoints, 1) * C;

% Initialize offset, alphas and gradient of objective function:
alphas = zeros(nPoints, 1);
objGrad = ones(nPoints, 1);
beta0 = NaN;  % Note that [1] uses a different sign convention here.

% These quantities are useful for the working set selection, see equations
% (7) and (11) of [3].
Avec = zeros(size(y));
Avec(y==-1) = - boxConstraints(y==-1);
Bvec = zeros(size(y));
Bvec(y==1) = boxConstraints(y==1);
upMask = y .* alphas < (Bvec - svTol);
downMask = y .* alphas > (Avec + svTol);
idxHelper = 1:nPoints;

% -------------------------------------
% MAIN LOOP
% -------------------------------------

% The main loop finds a pair of alphas as working set using the maximum
% gain method, see WSS3 in [2].

for itCount = 1 : maxIterations

    % Find the first alpha
    [val1, idx1] = max(y(upMask) .* objGrad(upMask));
    tmp = idxHelper(upMask);
    idx1 = tmp(idx1);

     % Find the second alpha according to the 'maximum violating pair' rule
    [val2,idx2] = min(y(downMask) .* objGrad(downMask));
    tmp = idxHelper(downMask);
    idx2 = tmp(idx2);
    
    % Stopping condition
    if val1 - val2 <= tolKKT
        beta0 = (val1 + val2) / 2;  %% see eq. (11) and (14) of [3]
        if verbose
            disp(getString(message('stats:svmtrain:SMOFinished')));
            reportStatus(itCount, val1 - val2);
        end
        return;
    end
    
    % We now have the working set. Next do the analytical solution:
    updateAlphas(idx1, idx2);
        
    % Print progress in verbose mode 2:
    if verbose == 2
        if mod(itCount, 500) == 0
            val1 = max(y(upMask) .* objGrad(upMask));
            val2 = min(y(downMask) .* objGrad(downMask));
            reportStatus(itCount, val1 - val2);
        end
    end
    
end % end of itCount loop

% Error exit because we never reached the right exit conditions
% error(message('stats:svmtrain:NoConvergence'))

% No convergence but we want to compute the offset anyway
beta0 = (val1 + val2) / 2;
   

% -------------------------------------
% HELPER FUNCTIONS
% -------------------------------------

    function updateAlphas(i, j)
        % This function calculates new values for alpha_i and alpha_j
        % according to eqs. (16) and (18) of [1].
        % j corresponds to index 2 and i corresponds to index 1.

        % Get relevant kernel matrix elements: eq. (15) of [1].
        eta = K(i,i) + K(j,j) - 2 * K(i,j);

        % Calculate clip limits: eq. (13) and (14) of [1].
        if y(i) * y(j) == 1
            Low = max(0, alphas(j) + alphas(i) - boxConstraints(i));
            High = min(boxConstraints(j), alphas(j) + alphas(i));
        else
            Low = max(0, alphas(j) - alphas(i));
            High = min(boxConstraints(j), boxConstraints(i) + alphas(j) - alphas(i));
        end

        if eta > eps
            % Calculate new values for alpha(i) and alpha(j) (but don't store
            % them yet in the global alpha array). This corresponds to
            % finding the right orientation of the separating plane,
            % eq. (16) of [1].
            lambda = - y(i) * objGrad(i) + y(j) * objGrad(j);
            alpha_j = alphas(j) + y(j) / eta * lambda;
            % Clip alpha: eq. (17) of [1].
            if alpha_j < Low
                alpha_j = Low;
            elseif alpha_j > High
                alpha_j = High;
            end
        else
            % The case 'eta < eps' should not happen too often (only for duplicate
            % data points and illegal kernels)! In this case we do
            % the ugly calculation of eq. (19) in [1]
            [psi_l, psi_h,Low,High] = evalPsiAtEnd(i,j, Low, High);
            if psi_l < (psi_h - eps)
                alpha_j  = Low;
            elseif psi_l > (psi_h + eps)
                alpha_j = High;
            else % no progress :-(
                alpha_j = alphas(j);
            end
        end

        % New value for alpha(i): eq. (18) of [1]
        alpha_i = alphas(i) + y(j) * y(i) * ...
            (alphas(j) - alpha_j);

        % We do have to make sure that the new alpha definitely
        % sits in its box.
        if alpha_i < eps
            alpha_i = 0;
        elseif alpha_i > (boxConstraints(i) - eps)
            alpha_i = boxConstraints(i);
        end

        % Update gradient of objective function:
        objGrad = objGrad - ...
            (K(:,i) .* y) * ...
            (alpha_i - alphas(i)) * y(i) - ...
            (K(:,j) .* y) * ...
            (alpha_j - alphas(j)) * y(j);

        % Update up and down masks:
        upMask(i) = y(i) * alpha_i < (Bvec(i) - svTol);
        downMask(i) = y(i) * alpha_i > (Avec(i) + svTol);
        upMask(j) = y(j) * alpha_j < (Bvec(j) - svTol);
        downMask(j) = y(j) * alpha_j > (Avec(j) + svTol);

        % Finally, update global alpha array
        alphas(i) = alpha_i;
        alphas(j) = alpha_j;
    end % end of updateAlphas

    function [psi_l, psi_h,Low,High] = evalPsiAtEnd(i, j, Low, High)
        % This function evaluates the objective function at the ends of
        % the feasible region. This is necessary in the hopefully rare
        % case of eta < 0. It looks a bit ugly but simply implements eq. (19) of [1].
        
        Kii = K(i, i);
        Kjj = K(j, j);
        Kij = K(i, j);

        s = y(i) * y(j);
        fi = - objGrad(i) - alphas(i) * Kii - s * alphas(j) * Kij;
        fj = - objGrad(j) - alphas(j) * Kjj - s * alphas(i) * Kij;
        Li = alphas(i) + s * (alphas(j) - Low);
        Hi = alphas(i) + s * (alphas(j) - High);
        psi_l = Li * fi + Low * fj + Li * Li * Kii / 2 + Low * Low * Kjj / 2 + ...
            s * Low * Li * Kij;
        psi_h = Hi * fi + High * fj + Hi * Hi * Kii / 2 + High * High * Kjj / 2 + ...
            s * High * Hi * Kij;
    end

    function flags = checkKKT()
        % This function checks which alphas satisfy the KKT conditions.
        % The return value is a logical array of the same size as alphas
        % and indicates which alphas satisfy the KKT conditions.

        amount = - objGrad + y * beta0;
        flags = false(size(amount));

        % SV: check y_i * u_i == 1
        freeSVmask = (alphas > svTol) & (alphas < (boxConstraints - svTol));
        flags(freeSVmask) = (abs(amount(freeSVmask)) < tolKKT);

        % Lower bound alphas: correctly classified and not an SV:
        % check u_i * y_i >= 1
        mask = alphas < svTol;
        flags(mask) = amount(mask) > -tolKKT;

        % Upper bound alphas: margin violators:
        % check u_i * y_i <= 1
        mask = (boxConstraints - alphas) < svTol;
        flags(mask) = amount(mask) <= tolKKT;
    end

    function reportStatus(numIter, stopCrit)
        % This function reports the current status of the algorithm and is called
        % only in verbose mode.

        % Calculate value of objective function
        objFunc = sum(alphas) + 1/2 * (objGrad'-1) * alphas;
        
        fprintf('%s',getString(message('stats:svmtrain:NumberOfIterations', numIter)));
        fprintf('Number of bound support vectors: %d\n', length(alphas(alphas==C)));
        fprintf('Number of essential support vectors: %d\n', length(alphas(alphas>0 & alphas<C)));
        fprintf('%s',getString(message('stats:svmtrain:ValueOfStoppingCriterion', sprintf('%f',stopCrit))));
        fprintf('%s',getString(message('stats:svmtrain:ValueOfObjectiveFunction', sprintf('%f',objFunc))));
        disp('---------------------------');
    end


end  % end of seqMinOptImpl


