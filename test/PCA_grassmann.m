function PCA_grassmann()
    clc; 
    close all;
    clear;
    
    rng('default');
    rng(0); 
    
    %% generate data    
    N = 100000; d = 200; r = 5;        
    fprintf('generating data ... ');
    x_sample = (randn(d, N)'*diag([[20 10 5 3 2], ones(1, d-5)]))';
    fprintf('done.\n');
    
    data.x = mat2cell(x_sample, d, ones(N, 1)); %     
 
    
    %% Obtain solution
    coeff = pca(x_sample');
    x_star = coeff(:,1:r);
    f_sol = -0.5/N*norm(x_star'*x_sample, 'fro')^2;
    fprintf('f_sol: %.16e, cond = %.2f\n', f_sol, 0);%cond(x_sample));
    
    
    %% Set manifold
    problem.M = grassmannfactory(d, r);
    problem.ncostterms = N;
    problem.d = d;    
    problem.data = data;    
    
    %% Define problem definitions
    problem.cost = @cost;
    function f = cost(U)
        f = -0.5*norm(U'*x_sample, 'fro')^2;
        f = f/N;
    end
    
    problem.egrad = @egrad;
    function g = egrad(U)
        g = - x_sample*(x_sample'*U);        
        g = g/N;
    end
    
    problem.partialegrad = @partialegrad;
    function g = partialegrad(U, indices)
        len = length(indices);
        x_sample_batchsize = x_sample(:,indices);        
        g = - x_sample_batchsize*(x_sample_batchsize'*U);
        g = g/len;
    end
    
        
    %% Run algorithms    
    
    % Initialize
    Uinit = problem.M.rand();
    
    % SD, CD
    max_iter = 20;  
    maxepoch = 45;
    tolgradnorm = 1e-10;
    
    % SGD
    batchsize_sgd = floor(N^(1/2));
    maxiter_sgd = batchsize_sgd * 40;
    checkperiod_sgd = 40;
    
    % SVRG, SRG
    batchsize_svrg = floor(N^(1/2));
    maxinneriter_svrg = floor(N^(1/2));
    batchsize_srg = floor(N^(1/2));
    maxinneriter_srg = floor(N^(1/2));    
    
    
    % RSD
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;         
    [~, ~, infos_rsd, ~] = steepestdescent_mod(problem, Uinit, options); 
    
    
    % RCG
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;    
    [~, ~, infos_rcg, ~] = conjugategradient_mod(problem, Uinit, options);
    
    
    % RSGD
    options.batchsize = batchsize_sgd; 
    options.maxiter = maxiter_sgd ; 
    options.checkperiod = checkperiod_sgd;
    options.verbosity = 1;
    options.stepsize_type = 'decay'; % 1/sqrt(t) decay.
    options.stepsize_init = 0.001;
    options.stepsize_lambda = 1e-2;
    options.transport = 'ret_vector';
    [~, infos_rsgd] = RSGD(problem, Uinit, options);   
    
    
    % RSRG
    clear options
    options.batchsize = batchsize_srg;   
    options.maxepoch = max_iter;
    options.maxinneriter = maxinneriter_srg;
    options.verbosity = 1;
    options.stepsize_type = 'fix'; 
    options.stepsize = 0.003;
    options.gamma = 0;
    options.transport = 'ret_vector'; 
    [~, infos_rsrg,~] = RSRG(problem, Uinit, options);
    
    % RSVRG
    clear options;    
    options.batchsize = batchsize_svrg;
    options.maxepoch = max_iter;
    options.maxinneriter = maxinneriter_svrg;
    options.verbosity = 1;
    options.stepsize_type = 'fix'; 
    options.stepsize = 0.003;
    options.transport = 'ret_vector'; 
    [~, infos_rsvrg,~] = RSVRG(problem, Uinit, options); 

    
    % R-AbaSRG
    clear options
    options.batchsize = batchsize_srg;
    options.maxepoch = max_iter;
    options.maxinneriter = maxinneriter_srg;
    options.verbosity = 1;
    options.stepsize_type = 'fix'; 
    options.stepsize = 0.003; 
    options.gamma = 0;
    options.transport = 'ret_vector'; 
    options.adapt_inneriter = 1;
    options.adapt_mbatchsize = 1;
    options.c = 6e5; 
    options.beta_1 = options.c/50; 
    [~, infos_adarsrg,~] = AdaRSRG(problem, Uinit, options);
    
    
    % R-AbaSVRG
    clear options
    options.batchsize = batchsize_svrg;
    options.maxepoch = max_iter;
    options.maxinneriter = maxinneriter_svrg;
    options.verbosity = 1;
    options.stepsize_type = 'fix'; 
    options.stepsize = 0.003;
    options.transport = 'ret_vector'; 
    options.adapt_inneriter =1;
    options.adapt_mbatchsize = 1;
    options.c = 6e5; 
    options.beta_1 = options.c/50;
    [~, infos_adarsvrg,~] = AdaRSVRG(problem, Uinit, options);

    
    % RSPIDER
    clear options
    options.batchsizeS1 = N;
    options.batchsizeS2 = batchsize_srg;
    options.maxepoch = max_iter*maxinneriter_srg;
    options.innerloop = maxinneriter_srg;
    options.checkperiod = maxinneriter_srg;
    options.verbosity = 1;
    options.stepsize_type = 'adaptive'; 
    options.stepsize_alpha = 0.1;
    options.stepsize_beta = 0.5; 
    options.transport = 'ret_vector'; 
    [~, infos_rspider,~] = RSPIDER(problem, Uinit, options); 
    
    %% Plots
    
    num_grads_rsd = (1:length([infos_rsd.cost])) - 1; % N*options_sd.maxiter;
    num_grads_rcg = (1:length([infos_rcg.cost])) - 1; % N*options_sd.maxiter;
    
    optgap_rsd = abs([infos_rsd.cost] - f_sol);
    optgap_rcg = abs([infos_rcg.cost] - f_sol);
    outgap_rsgd = abs([infos_rsgd.cost] - f_sol);
    outgap_rspider = abs([infos_rspider.cost] - f_sol);
    optgap_rsrg = abs([infos_rsrg.cost] - f_sol);
    optgap_rsvrg = abs([infos_rsvrg.cost] - f_sol);
    optgap_adarsrg = abs([infos_adarsrg.cost] - f_sol);
    outgap_adarsvrg = abs([infos_adarsvrg.cost] - f_sol);
    

    fs = 18;  
    fs_ax = 16;
    lw = 3;
    set(0, 'DefaultAxesFontName', 'ArialMT'); 
            
    % Optimality gap vs IFO calls
    figure();
    plot(221)
    semilogy(num_grads_rsd, optgap_rsd, '-', 'LineWidth', lw);  hold on;
    semilogy(num_grads_rcg, optgap_rcg, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_rsgd.gradcnt]/N, outgap_rsgd, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_rspider.gradcnt]/N, outgap_rspider, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_rsrg.gradcnt]/N, optgap_rsrg, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_rsvrg.gradcnt]/N, optgap_rsvrg, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_adarsrg.gradcnt]/N, optgap_adarsrg, '-.', 'LineWidth', lw);  hold on;
    semilogy([infos_adarsvrg.gradcnt]/N, outgap_adarsvrg, '-.', 'LineWidth', lw);  hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize', fs_ax, 'FontWeight','bold');
    xlim([0 40]);
    ylim([1e-9 1e4]);
    yt=arrayfun(@num2str,log10(get(gca,'ytick')), 'un',0); % change into log scale
    set(gca,'yticklabel',yt);
    xlabel('IFO/n', 'fontsize', fs,'FontWeight','bold');
    ylabel('Optimality gap (log)', 'fontsize', fs, 'FontWeight', 'bold');
    legend('R-SD', 'R-CG', 'R-SGD', 'R-SPIDER', 'R-SRG', 'R-SVRG', 'R-AbaSRG', 'R-AbaSVRG');
    
    
    % Optimality gap vs time
    figure();
    plot(221)
    semilogy([infos_rsd.time], optgap_rsd, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_rcg.time], optgap_rcg, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_rsgd.time], outgap_rsgd, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_rspider.time], outgap_rspider, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_rsrg.time], optgap_rsrg, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_rsvrg.time], optgap_rsvrg, '-', 'LineWidth', lw);  hold on;
    semilogy([infos_adarsrg.time], optgap_adarsrg, '-.', 'LineWidth', lw);  hold on;
    semilogy([infos_adarsvrg.time], outgap_adarsvrg, '-.', 'LineWidth', lw);  hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize', fs_ax, 'FontWeight','bold');
    xlim([0 6]);
    ylim([1e-8 1e4]);
    yt=arrayfun(@num2str,log10(get(gca,'ytick')), 'un',0); % change into log scale
    set(gca,'yticklabel',yt);
    xlabel('Time (s)', 'fontsize', fs,'FontWeight','bold');
    ylabel('Optimality gap (log)', 'fontsize', fs, 'FontWeight', 'bold');
    legend('R-SD', 'R-CG', 'R-SGD','R-SPIDER', 'R-SRG', 'R-SVRG', 'R-AbaSRG', 'R-AbaSVRG');
    
    
end
