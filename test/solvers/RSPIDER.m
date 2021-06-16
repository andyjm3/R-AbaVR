function [x, info, options] = RSPIDER(problem, x, options)
% The solver is based on the paper 
%@inproceedings{zhou2019faster,
%  title={Faster first-order methods for stochastic non-convex optimization on Riemannian manifolds},
%  author={Zhou, Pan and Yuan, Xiao-Tong and Feng, Jiashi},
%  booktitle={The 22nd International Conference on Artificial Intelligence and Statistics},
%  pages={138--147},
%  year={2019},
%  organization={PMLR}
%}

% Original authors: Andi Han, July, 2020.
 
    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
            'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
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
    localdefaults.maxepoch = 100;  % Maximum number of epochs.
    localdefaults.innerloop = 10; % innerloop frequency
    localdefaults.stepsize = 0.01;  % Initial stepsize guess.
    localdefaults.stepsize_type = 'adaptive'; % Stepsize type. Other possibilities are 'fix' and 'adaptive'.
    localdefaults.stepsize_alpha = 0.1; % alpha is a weighting factor while using stepsize_typ='adaptive'.
    localdefaults.stepsize_beta = 0.01; % beta is a weighting factor while using stepsize_typ = 'adaptive'
    localdefaults.stepsize_lambda = 0.1; 
    localdefaults.tolgradnorm = 1.0e-10; % Batch grad norm tolerance.
    localdefaults.batchsizeS1 = N;  % Batchsize S1
    localdefaults.batchsizeS2 = sqrt(N); % Batchsize S2
    localdefaults.verbosity = 1;  % Output verbosity. Other localdefaults are 1 and 2.
    localdefaults.transport = 'ret_vector';
    
    % Check stopping criteria and save stats every checkperiod iterations.
    localdefaults.checkperiod = 100; 
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    alpha = options.stepsize_alpha;
    beta = options.stepsize_beta;
    batchsizeS1 = options.batchsizeS1;
    batchsizeS2 = options.batchsizeS2;
    maxepoch = options.maxepoch;
    stepsize0 = options.stepsize;
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = problem.M.norm(x, grad);
    grad_cnt = 0;
    
    % Total number of saved stats at this point.
    savedstats = 0;
    current_iter = 0; %total iteration
    
    % Collect and save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    savedstats = savedstats + 1;
    preallocate = maxepoch + 1;
    
    info(preallocate).iter = [];
    
    mode = 'RSPIDER';
    
    % we only display info at check point
    if options.verbosity > 0
        fprintf('\n-------------------------------------------------------\n');
        fprintf('%s:  iter\t               cost val\t    grad. norm\t stepsize\n', mode);
        fprintf('%s:  %5d\t%+.16e\t%.8e\t%.8e\n', mode, 0, cost, gradnorm,stepsize0);
    end
    
    
    % Main loop over epoch.
    elapsed_time = 0;
    for epoch = 1 : options.maxepoch
        
        start_time = tic;
        
        if mod(epoch-1, options.innerloop) == 0
            idx_batchS1 = randsample(N, batchsizeS1, false);
            partialgrad = getPartialGradient(problem, x, idx_batchS1, storedb, key);
            
            % Increment grad_cnt for grad S1
            grad_cnt = grad_cnt + batchsizeS1;
        else
            idx_batchS2 = randi(N, 1, batchsizeS2);
            partialgrad = getPartialGradient(problem, x, idx_batchS2, storedb, key);
            partialgrad_prev = getPartialGradient(problem, x_prev, idx_batchS2);
            
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
            
            partialgrad = problem.M.lincomb(x, 1, v_trans, 1, partialgrad);
            partialgrad = problem.M.lincomb(x, 1, partialgrad, -1, partialgrad_prev_trans);
            
            % Increment grad_cnt for grad S2
            grad_cnt = grad_cnt + 2*batchsizeS2;
            
        end
        
        % Update Stepsize
        if strcmp(options.stepsize_type, 'adaptive')
            stepsize = alpha^(floor(epoch/options.innerloop)) * beta;

        elseif strcmp(options.stepsize_type, 'fix')
            stepsize = stepsize0; % Fixed stepsize.
            
        elseif strcmp(options.stepsize_type, 'decay')
            stepsize = stepsize0 / (1  + stepsize0 * options.stepsize_lambda * epoch);
        else
            error(['Unknown options.stepsize_type. ' ...
                'Should be fix or decay.']);
        end
        
        
        % Update x 
        x_prev = x;
        vk_gradnorm = problem.M.norm(x, partialgrad);
        if strcmp(options.transport, 'exp_parallel') && isfield(problem.M, 'paratransp')           
            x =  problem.M.exp(x, partialgrad, -stepsize/vk_gradnorm);
        elseif strcmp(options.transport, 'ret_vector_locking') && isfield(problem.M, 'transp_locking')                
            x =  problem.M.exp(x, partialgrad, -stepsize/vk_gradnorm);
        else
            x =  problem.M.retr(x, partialgrad, -stepsize/vk_gradnorm);
        end
        newkey = storedb.getNewKey(); 
        key = newkey;
        
        v_prev = partialgrad;
        move = - partialgrad * stepsize/vk_gradnorm;
        Estepsize = stepsize/vk_gradnorm;
               
        
        elapsed_time = elapsed_time + toc(start_time);
        
        if mod(epoch, options.checkperiod) == 0
                
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
            
    end
    info = info(1:savedstats);
    
    
    function stats = savestats()
        stats.iter = current_iter;
        
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
            stats.Estepsize = NaN;
        else
            stats.time = info(savedstats).time + elapsed_time;
            stats.stepsize = stepsize;
            stats.cost = cost;
            stats.gradnorm = gradnorm;
            stats.gradcnt = grad_cnt;
            stats.Estepsize = Estepsize;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    
end

