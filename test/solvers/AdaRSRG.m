function [x, info, options] = AdaRSRG(problem, x, options)
% The solver is based on the paper
%@article{han2021improved,
%  title={Improved Variance Reduction Methods for Riemannian non-Convex Optimization},
%  author={Han, Andi and Gao, Junbin},
%  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
%  year={2021},
%  publisher={IEEE}
%}
 
% Original authors: Andi Han, July, 2020.

    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
            'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate Hessian is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
            ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    
    if ~canGetPartialGradient(problem)
        warning('manopt:getPartialGradient', ...
            'No partial gradient provided. The algorithm will likely abort.');
    end
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    
    
    % Total number of samples
    N = problem.ncostterms;
    
    % Set local defaults
    localdefaults.maxepoch = 20;  % Maximum number of epochs.
    localdefaults.maxinneriter = 100;  % Maximum number of sampling per epoch.
    localdefaults.stepsize = 0.001;  % Initial stepsize guess.
    localdefaults.stepsize_type = 'fix'; % Stepsize type. Other possibilities are 'fix' and 'decay'.
    localdefaults.stepsize_lambda = 0.05; % lambda is a weighting factor while using stepsize_typ='decay'.
    localdefaults.tolgradnorm = 1.0e-10; % Batch grad norm tolerance.
    localdefaults.batchsize = 50;  % Batchsize.
    localdefaults.verbosity = 1;  % Output verbosity. Other localdefaults are 1 and 2.
    localdefaults.gamma = 0; % Threshold parameter for R-SRG+.
    localdefaults.transport = 'ret_vector';
    
    % ada
    localdefaults.c = 10000; % c_beta * sigma in paper
    localdefaults.beta_1 = max(localdefaults.maxepoch * 0.0001, 1);
    localdefaults.adapt_inneriter = 0; % whether to adapt inner loop size to the same as batch size
    localdefaults.adapt_mbatchsize = 0; % whether to adapt mini-batch size to the same as batch size
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
        
    stepsize0 = options.stepsize;
    batchsize = options.batchsize;
    maxepoch = options.maxepoch;
    maxinneriter = options.maxinneriter;
    
    % ada
    c_beta = options.c;
    beta_1 = options.beta_1;
    adapt_inneriter = options.adapt_inneriter;
    adapt_mbatchsize = options.adapt_mbatchsize;

    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = problem.M.norm(x, grad);
    
    % Total number of saved stats at this point.
    savedstats = 0;
    grad_cnt = 0;
    epoch = 0;
    
    % Collect and save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    savedstats = savedstats + 1;
    preallocate = maxepoch*maxinneriter + 1;
    
    info(preallocate).iter = [];
    
    if options.gamma > 0
        mode = 'R-AdaSRG(+)';
    else
        mode = 'R-AdaSRG';
    end
    
    
    if options.verbosity > 0
        fprintf('\n-------------------------------------------------------\n');
        fprintf('%s:  epoch\t               cost val\t    grad. norm\t stepsize\n', mode);
        fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e\n', mode, 0, cost, gradnorm,stepsize0);
    end    
    
    
    % Main loop over epoch.
    iter = 0;           % for step size decay
    elapsed_time = 0;   
    beta_s = beta_1;
    adarun = 1;         % check if adaptive is necessary
    Ns_prev = 0;        % track previous batch size
    for epoch = 1 : options.maxepoch
        
        start_time = tic;
        
        % Determine whether batch size has reached N, then stop adapt
        if Ns_prev == N
            adarun = 0;
            Ns = N;
        end
        % Update batchsize if have not reahced N
        if adarun
            Ns = ceil(min(c_beta/beta_s , N)); 
            idx_out = randsample(N, Ns, false); % sample without replacement
            grad0 = getPartialGradient(problem, x, idx_out, storedb, key);
        else
            [~, grad0] = getCostGrad(problem, x, storedb, key);
        end
        
        % Update stepsize
        if strcmp(options.stepsize_type, 'decay')
            stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).

        elseif strcmp(options.stepsize_type, 'fix')
            stepsize = stepsize0; % Fixed stepsize.

        else
            error(['Unknown options.stepsize_type. ' ...
                'Should be fix or decay.']);
        end
        
        
        % Update x with full gradient
        x_prev = x;
        if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')           
            x =  problem.M.exp(x, grad0, -stepsize);
        elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')                
            x =  problem.M.exp(x, grad0, -stepsize);
        else
            x =  problem.M.retr(x, grad0, -stepsize);
        end
        newkey = storedb.getNewKey(); 
        key = newkey;
        
        % Store 
        v_prev = grad0;
        move   = - grad0 * stepsize; 
        
        if options.gamma > 0
            grad0_norm = problem.M.norm(x_prev, v_prev);
        end 
        
        % Increment grad_cnt and elapsed time for epoch
        grad_cnt = grad_cnt + Ns;
        elapsed_time = elapsed_time + toc(start_time);
        
        % Per epoch: main inner loop.
        break_inneriter = 0;
        if adapt_inneriter && adarun        % if adapt inner loop size
            if Ns >= options.maxinneriter
                maxinneriter = options.maxinneriter;
            else
                maxinneriter = Ns;
            end
        end
        if adapt_mbatchsize && adarun      % if adapt mini-batch size
            if Ns >= options.batchsize
                batchsize = options.batchsize;
            else
                batchsize = Ns;
            end
        end
        if adarun
            beta_s = problem.M.norm(x, grad0)^2/maxinneriter;
        end
        perm_idx = randi(N, 1, maxinneriter * batchsize); % Draw the samples with replacement for inner loop
        for inneriter = 1 : maxinneriter
            
            % Set start time
            start_time = tic;
            
            % Pick a sample of size batchsize
            start_index = (inneriter - 1)* batchsize + 1;
            end_index = inneriter * batchsize;
            idx_batchsize = perm_idx(start_index : end_index);
            
            % Compute the gradient on this batch.
            partialgrad = getPartialGradient(problem, x, idx_batchsize, storedb, key);
            partialgrad_prev = getPartialGradient(problem, x_prev, idx_batchsize);            
                        
            % Update stepsize
            if strcmp(options.stepsize_type, 'decay')
                stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
                
            elseif strcmp(options.stepsize_type, 'fix')
                stepsize = stepsize0; % Fixed stepsize.
                
            else
                error(['Unknown options.stepsize_type. ' ...
                    'Should be fix or decay.']);
            end
            
            % Update partialgrad
            if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')
                % parallel translation
                
                v_trans = problem.M.paratransp(x_prev, move, v_prev);  
                partialgrad_prev_trans = problem.M.paratransp(x_prev, move, partialgrad_prev);  
                
            elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')

                v_trans = problem.M.transp_locking(x_prev, move, x, v_prev); 
                partialgrad_prev_trans = problem.M.transp_locking(x_prev, move, x, partialgrad_prev);                  

            else
                % Vector transport
                
                v_trans = problem.M.transp(x_prev, x, v_prev); 
                partialgrad_prev_trans = problem.M.transp(x_prev, x, partialgrad_prev);
            end

            % Update partialgrad to reduce variance by
            % taking a linear combination with old gradients.
            % We make the combination
            % partialgrad + partialgrad_prev_trans - v_trans.
            partialgrad = problem.M.lincomb(x, 1, v_trans, 1, partialgrad);
            partialgrad = problem.M.lincomb(x, 1, partialgrad, -1, partialgrad_prev_trans);

            % Update x
            if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')           
                xnew =  problem.M.exp(x, partialgrad, -stepsize);
            elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')                
                xnew =  problem.M.exp(x, partialgrad, -stepsize);
            else
                xnew =  problem.M.retr(x, partialgrad, -stepsize);
            end
            newkey = storedb.getNewKey();
            
            iter = iter + 1; % Total number updates.
            
            if adarun
                beta_s = beta_s + problem.M.norm(x,partialgrad)^2/maxinneriter; 
            end
            
            elapsed_time = elapsed_time + toc(start_time);
                        
            % store variable
            x_prev  = x;  
            v_prev  = partialgrad;
            move    = - partialgrad * stepsize;
            
            x = xnew;
            key = newkey;
            
            % Increment grad_cnt for stochstic grad
            grad_cnt = grad_cnt + 2*batchsize;
            
            if options.gamma > 0
                v_prev_norm = problem.M.norm(x_prev, v_prev);

                if (v_prev_norm <  options.gamma*grad0_norm) && inneriter > 1
                    break_inneriter = inneriter;
                    break;
                end
            end 
            
        end
        
        Ns_prev = Ns;
        
        % Save after every inner loop
        [cost, newgrad] = getCostGrad(problem, x, storedb, key);
        gradnorm = problem.M.norm(x, newgrad);

        % Log statistics for freshly executed iteration.
        stats = savestats();
        info(savedstats+1) = stats;
        savedstats = savedstats + 1;

        % Reset timer.
        elapsed_time = 0;

        % Print output.
        if options.verbosity > 0
            if break_inneriter == 0
                fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e\n', mode, epoch, cost, gradnorm, stepsize);
            else
                fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e (break at %5d)\n', mode, epoch, cost, gradnorm, stepsize, break_inneriter);
            end
        end

        if gradnorm  <= options.tolgradnorm
            if options.verbosity > 0
                fprintf('\nNorm of gradient smaller than %g.\n',options.tolgradnorm);
            end
        end
        
        
    end
    
    info = info(1:savedstats);
    
    
    function stats = savestats()
        stats.iter = epoch;
        
        % Compute partial Riemannian gradient on this batch.
        %grad = getPartialGradient(problem, x, [1:problem.ncostterms], storedb, key);
        %gradnorm = problem.M.norm(x, grad);
        %stats.gradnorm = gradnorm;
        
        if savedstats == 0
            stats.time = 0;
            stats.stepsize = NaN;
            stats.cost = cost;
            stats.gradnorm = gradnorm;
            stats.gradcnt = grad_cnt;
            stats.batchsize = NaN;
            stats.maxinneriter = NaN;
            %stats.mbatchsize = NaN;
        else
            stats.time = info(savedstats).time + elapsed_time;
            stats.stepsize = stepsize;
            stats.cost = cost;
            stats.gradnorm = gradnorm;
            stats.gradcnt = grad_cnt;
            stats.batchsize = Ns;
            stats.maxinneriter = maxinneriter;
            %stats.mbatchsize = batchsize;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    

end


