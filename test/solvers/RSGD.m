function [x, info, options] = RSGD(problem, x, options)
% Stochastic gradient (SG) minimization algorithm for Manopt.
%
% function [x, info, options] = stochasticgradient(problem)
% function [x, info, options] = stochasticgradient(problem, x0)
% function [x, info, options] = stochasticgradient(problem, x0, options)
% function [x, info, options] = stochasticgradient(problem, [], options)
%
% Apply the Riemannian stochastic gradient algorithm to the problem defined
% in the problem structure, starting at x0 if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x0 as [] (the empty matrix).
%
% The problem structure must contain the following fields:
%
%  problem.M:
%       Defines the manifold to optimize over, given by a factory.
%
%  problem.partialgrad or problem.partialegrad (or equivalent)
%       Describes the partial gradients of the cost function. If the cost
%       function is of the form f(x) = sum_{k=1}^N f_k(x),
%       then partialegrad(x, K) = sum_{k \in K} grad f_k(x).
%       As usual, partialgrad must define the Riemannian gradient, whereas
%       partialegrad defines a Euclidean (classical) gradient which will be
%       converted automatically to a Riemannian gradient. Use the tool
%       checkgradient(problem) to check it.
%
%  problem.ncostterms
%       An integer specifying how many terms are in the cost function (in
%       the example above, that would be N.)
%
% Importantly, the cost function itself needs not be specified.
%
% Some of the options of the solver are specific to this file. Please have
% a look inside the code.
%
% To record the value of the cost function or the norm of the gradient for
% example (which are statistics the algorithm does not require and hence
% does not compute by default), one can set the following options:
%
%   metrics.cost = @(problem, x) getCost(problem, x);
%   metrics.gradnorm = @(problem, x) problem.M.norm(x, getGradient(problem, x));
%   options.statsfun = statsfunhelper(metrics);
%
% Important caveat: stochastic algorithms usually return an average of the
% last few iterates. Computing averages on manifolds can be expensive.
% Currently, this solver does not compute averages and simply returns the
% last iterate. Using options.statsfun, it is possible for the user to
% compute averages manually. If you have ideas on how to do this
% generically, we welcome feedback. In particular, approximate means could
% be computed with M.pairmean which is available in many geometries.
%
% See also: steepestdescent

% This file is part of Manopt: www.manopt.org.
% Original authors: Bamdev Mishra <bamdevm@gmail.com>,
%                   Hiroyuki Kasai <kasai@is.uec.ac.jp>, and
%                   Hiroyuki Sato <hsato@ms.kagu.tus.ac.jp>, 22 April 2016.
% Contributors: Nicolas Boumal
% Change log: 
    

    % Verify that the problem description is sufficient for the solver.
    if ~canGetPartialGradient(problem)
        warning('manopt:getPartialGradient', ...
         'No partial gradient provided. The algorithm will likely abort.');
    end
    
   
    % Set local default
    localdefaults.maxiter = 1000;       % Maximum number of iterations
    localdefaults.batchsize = 1;        % Batchsize (# cost terms per iter)
    localdefaults.verbosity = 2;        % Output verbosity (0, 1 or 2)
    localdefaults.storedepth = 20;      % Limit amount of caching
    localdefaults.stepsize_init = 0.01;
    localdefaults.stepsize_lambda = 0.01; 
    
    % Check stopping criteria and save stats every checkperiod iterations.
    localdefaults.checkperiod = 100;
  
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    
    assert(options.checkperiod >= 1, ...
                 'options.checkperiod must be a positive integer (>= 1).');
    
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    stepsize0 = options.stepsize_init;
    mode = 'RSGD';
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    
    % Elapsed time for the current set of iterations, where a set of
    % iterations comprises options.checkperiod iterations. We do not
    % count time spent for such things as logging statistics, as these are
    % not relevant to the actual optimization process.
    elapsed_time = 0;
    
    % Total number of completed steps
    iter = 0;
        
    
    %
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = problem.M.norm(x, grad);
    grad_cnt = 0; 
    savedstats = 0;
    
    % Collect and save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    savedstats = savedstats + 1;
    if isinf(options.maxiter)
        % We trust that if the user set maxiter = inf, then they defined
        % another stopping criterion.
        preallocate = 1e5;
    else
        preallocate = ceil(options.maxiter / options.checkperiod) + 1;
    end
    info(preallocate).iter = [];
        
    
    % Display information header for the user.
    if options.verbosity >= 1
        fprintf('\n-------------------------------------------------------\n');
        fprintf('%s:  iter\t               cost val\t    grad. norm\t stepsize\n', mode);
        fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e\n', mode, 0, cost, gradnorm,stepsize0);
    end
    
    
    % Main loop.
    stop = false;
    while iter < options.maxiter
        
        % Record start time.
        start_time = tic();
        
        % Draw the samples with replacement.
        idx_batch = randi(problem.ncostterms, options.batchsize, 1);
        
        % Compute partial gradient on this batch.
        pgrad = getPartialGradient(problem, x, idx_batch, storedb, key);
        
        grad_cnt = grad_cnt + options.batchsize;
        
        % Compute a step size and the corresponding new point x.
        %[stepsize, newx, newkey, ssstats] = ...
        %                   options.stepsizefun(problem, x, pgrad, iter, ...
        %                                       options, storedb, key);
        
        if strcmp(options.stepsize_type, 'decay')
            stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).

        elseif strcmp(options.stepsize_type, 'fix')
            stepsize = stepsize0; % Fixed stepsize.

        elseif strcmp(options.stepsize_type, 'hybrid')
            %if epoch < 5 % Decay stepsize only for the initial few epochs.
            if epoch < 3 % Decay stepsize only for the initial few epochs.              % HK      
                stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
            end

        else
            error(['Unknown options.stepsize_type. ' ...
                'Should be fix or decay.']);
        end
        
        if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')           
            newx =  problem.M.exp(x, pgrad, -stepsize);
        elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')                
            newx =  problem.M.exp(x, pgrad, -stepsize);
        else
            newx =  problem.M.retr(x, pgrad, -stepsize);
        end
        newkey = storedb.getNewKey(); 
        
        % Make the step.
        x = newx;
        key = newkey;
        
        % Total number of completed steps.
        iter = iter + 1;
        
        % Make sure we do not use too much memory for the store database.
        %storedb.purge();
        
        % Elapsed time doing actual optimization work so far in this
        % set of options.checkperiod iterations.
        elapsed_time = elapsed_time + toc(start_time);
        
        % Check stopping criteria and save stats every checkperiod iters.
        if mod(iter, options.checkperiod) == 0
            
            %compute gradnorm
            [cost, newgrad] = getCostGrad(problem, x, storedb, key);
            gradnorm = problem.M.norm(x, newgrad);
            
            % Log statistics for freshly executed iteration.
            stats = savestats();
            info(savedstats+1) = stats;
            savedstats = savedstats + 1;
            
            % Reset timer.
            elapsed_time = 0;
            
            % Print output.
            if options.verbosity >= 1
                fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e\n', mode, iter/options.checkperiod, cost, gradnorm, stepsize);
            end
            
            % Run standard stopping criterion checks.
            [stop, reason] = stoppingcriterion(problem, x, ...
                                               options, info, savedstats);
            if stop
                if options.verbosity >= 1
                    fprintf([reason '\n']);
                end
                break;
            end
        
        end

    end
    
    
    % Keep only the relevant portion of the info struct-array.
    info = info(1:savedstats);    
    
    
    % Helper function to collect statistics to be saved at
    % index checkperiodcount+1 in info.
    function stats = savestats()
        stats.iter = iter;
        if savedstats == 0
            stats.time = 0;
            stats.stepsize = NaN;
            stats.cost = cost;
            stats.gradnorm = gradnorm;
            stats.gradcnt = grad_cnt;
        else
            stats.time = info(savedstats).time + elapsed_time;
            stats.stepsize = stepsize;
            stats.cost = cost;
            stats.gradnorm = gradnorm;
            stats.gradcnt = grad_cnt;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    
end
