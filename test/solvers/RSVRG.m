function [x, info, options] = RSVRG(problem, x, options)
% The Riemannian SVRG/SGD algorithms.
%
% function [x, cost, info, options] = Riemannian_svrg(problem)
% function [x, cost, info, options] = Riemannian_svrg(problem, x)
% function [x, cost, info, options] = Riemannian_svrg(problem, x, options)
% function [x, cost, info, options] = Riemannian_svrg(problem, [], options)
%
% Apply the Riemannian SVRG/SGD algorithm to the problem defined
% in the problem structure, starting at x if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x as [] (the empty matrix).
%
% The solver mimics other solvers of Manopt with two additonal input
% requirements: problem.ncostterms and problem.partialegrad.
%
% problem.ncostterms has the number of samples, e.g., N samples.
%
% problem.partialegrad takes input a current point of the manifold and
% index of batchsize.
%
% Some of the options of the solver are specifict to this file. Please have
% a look below.
%
% The solver is based on the paper by
% S.Bonnabel, 
% "Stochastic gradient descent on Riemannian manifolds," IEEE Trans. on Auto. Cont., 2013.
% H. Kasai, H. Sato, and B. Mishra,
% "Riemannian stochastic variance reduced gradient on Grassmann manifold,"
% Technical report, arXiv preprint arXiv:1605.07367, 2016.
% "Riemannian stochastic variance reduced gradient,"
% Technical report, arXiv preprint arXiv:1702.05594, 2017.



% Original authors: Bamdev Mishra <bamdevm@gmail.com>,
%                   Hiroyuki Kasai <kasai@is.uec.ac.jp>, and
%                   Hiroyuki Sato <hsato@ms.kagu.tus.ac.jp>, 22 April 2016.
  
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
    localdefaults.stepsize_type = 'fix'; % Stepsize type. Other possibilities are 'fix' and 'hybrid'.
    localdefaults.stepsize_lambda = 0.05; % lambda is a weighting factor while using stepsize_typ='decay'.
    localdefaults.tolgradnorm = 1.0e-10; % Batch grad norm tolerance.
    localdefaults.batchsize = 50;  % Batchsize.
    localdefaults.verbosity = 1;  % Output verbosity. Other localdefaults are 1 and 2.
    localdefaults.boost = false;   % True: do a normal SGD at the first epoch when SVRG.
    localdefaults.update_type = 'svrg'; 
    localdefaults.svrg_type = 1;  % To implement both the localdefaults that are used to define x0.
    localdefaults.transport = 'ret_vector';    
    
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
    boost = options.boost;
    if boost
        mode = 'RSVRG+';
    else
        mode = 'RSVRG';
    end
        
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = problem.M.norm(x, grad);
    
    % Save stats in a struct array info, and preallocate.
    % Total number of saved stats at this point.
    savedstats = 0;
    grad_cnt = 0;
    epoch = 0; %total iteration
    
    % Collect and save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    savedstats = savedstats + 1;
    preallocate = maxepoch*maxinneriter + 1;
    
    info(preallocate).iter = [];
    
    % we only display info at check point
    if options.verbosity > 0
        fprintf('\n-------------------------------------------------------\n');
        fprintf('%s:  iter\t               cost val\t    grad. norm\t stepsize\n', mode);
        fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e\n', mode, 0, cost, gradnorm,stepsize0);
    end
    
    
    % Main loop over epoch.
    iter = 0;           % for step size calculation
    toggle = 0;         % To check boosting.
    elapsed_time = 0;
    for epoch = 1 : options.maxepoch
        
        start_time = tic;
        
        % Draw the samples with replacement.
        perm_idx = randi(N, 1, maxinneriter*batchsize);
        
        % Compute grad
        x0 = x;
        %[~, grad0] = getCostGrad(problem, x, storedb, key);
        %grad0 = getGradient(problem, x, storedb, key);
        grad0 = getPartialGradient(problem, x, [1:N], storedb, key);
        
        % Check if boost is required for svrg
        if strcmp(options.update_type, 'svrg') && options.boost && epoch == 1
            options.update_type = 'sgd';
            toggle = 1;
        end
        
        if strcmp(options.update_type, 'svrg') && options.svrg_type == 2
            update_instance = randi(totalbatches, 1) - 1; % pick a number uniformly between 0 to m - 1.
            if update_instance == 0
                xsave = x0;
                gradsave = grad0;
            end
        end
        
        % Increment grad_cnt for full grad
        grad_cnt = grad_cnt + N;
        elapsed_time = elapsed_time + toc(start_time);
        
        % Per epoch: main loop over samples.
        for inneriter = 1 : maxinneriter
            
            % Set start time
            start_time = tic;
                        
            % Pick a sample of size batchsize
            start_index = (inneriter - 1)* batchsize + 1;
            end_index = inneriter * batchsize;
            idx_batchsize = perm_idx(start_index : end_index);
            
            % Compute the gradient on this batch.
            partialgrad = getPartialGradient(problem, x, idx_batchsize, storedb, key);
            
            % Update stepsize
            if strcmp(options.stepsize_type, 'decay')
                stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
                
            elseif strcmp(options.stepsize_type, 'fix')
                stepsize = stepsize0; % Fixed stepsize.
                
            elseif strcmp(options.stepsize_type, 'hybrid')
                %if epoch < 5 % Decay stepsize only for the initial few epochs.
                if epoch < 50 % Decay stepsize only for the initial few epochs.              % HK      
                    stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * iter); % Decay with O(1/iter).
                end
                
            else
                error(['Unknown options.stepsize_type. ' ...
                    'Should be fix or decay.']);
            end
            
            
            % Update partialgrad
            if strcmp(options.update_type, 'svrg')
                
                if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')
                
                    % Logarithm map
                    logmapX0ToX = problem.M.log(x0, x);

                    % Parallel translate from U0 to U.
                    grad0_transported = problem.M.paratransp(x0, logmapX0ToX, grad0);

                    % Caclculate partialgrad at x0
                    partialgrad0 = getPartialGradient(problem, x0, idx_batchsize);                    

                    % Caclculate transported partialgrad from x0 to x
                    partialgrad0_transported = problem.M.paratransp(x0, logmapX0ToX, partialgrad0);  
                    
                elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')
                    
                    % Logarithm map
                    %logmapX0ToX = problem.M.log(x0, x);
                    %VecX0ToX = logmapX0ToX;
                    % Projection of ( x-x0 )
                    VecX0ToX = problem.M.proj(x0, x-x0);  
                    % vector transport from x0 to x.
                    grad0_transported = problem.M.transp_locking(x0, VecX0ToX, x, grad0);


                    % Caclculate partialgrad at x0
                    partialgrad0 = getPartialGradient(problem, x0, idx_batchsize);                    

                    % Caclculate transported partialgrad from x0 to x
                    partialgrad0_transported = problem.M.transp_locking(x0, VecX0ToX, x, partialgrad0);  
                
                    
                else
                    
                    % Caclculate transported full batch gradient from x0 to x.
                    grad0_transported = problem.M.transp(x0, x, grad0); % Vector transport.

                    % Caclculate partialgrad at x0
                    partialgrad0 = getPartialGradient(problem, x0, idx_batchsize);

                    % Caclculate transported partialgrad from x0 to x
                    partialgrad0_transported = problem.M.transp(x0, x, partialgrad0); % Vector transport.

                end
                
                
                
                % Update partialgrad to reduce variance by
                % taking a linear combination with old gradients.
                % We make the combination
                % partialgrad + grad0 - partialgrad0.
                partialgrad = problem.M.lincomb(x, 1, grad0_transported, 1, partialgrad);
                partialgrad = problem.M.lincomb(x, 1, partialgrad, -1, partialgrad0_transported);
                
                
            elseif strcmp(options.update_type, 'sgd')
                % Do nothing
                
            else
                error(['Unknown options.update_type. ' ...
                    'Should be svrg or sgd.']);
                
            end
            
            % Update x
            if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')           
                xnew =  problem.M.exp(x, partialgrad, -stepsize);
                
            elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')                
                xnew =  problem.M.exp(x, partialgrad, -stepsize);
                
            else
                xnew =  problem.M.retr(x, partialgrad, -stepsize);
            end
            newkey = storedb.getNewKey();
            
            
            % Increment grad_cnt for stochstic grad
            grad_cnt = grad_cnt + 2*batchsize;            
            iter = iter + 1; % Total number updates.
            
            if strcmp(options.update_type, 'svrg') && options.svrg_type == 2 && inneriter == update_instance
                xsave = xnew;
                gradsave = getGradient(problem, xnew);
            end
            
            x = xnew;
            key = newkey;
            
            % Elapsed time
            elapsed_time = elapsed_time + toc(start_time);
            
        end
        
        % Reset if boosting used already.
        if toggle == 1
            options.update_type = 'svrg';
        end
        
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
            fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e\n', mode, epoch, cost, gradnorm, stepsize);

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


