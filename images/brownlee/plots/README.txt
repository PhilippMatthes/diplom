[dataset]_MLP_metrics[_more]_objective1_objective2.pdf
 - plots showing the full set of results from the grid search. objective1 is either cross-fold validation accuracy or test data accuracy. objective2 is energy or time on training or testing. ("more" indicates that the larger set of hyperparameter values were used)

[dataset]_MLP_metrics_more_hls_testcpuenergy.pdf
[dataset]_MLP_metrics_more_hls_traingcpuenergy.pdf
 - relationship between hidden layer size and cpu energy, for training and testing

[dataset]_MLP_metrics_more_[training|testing]_cputime_vs_cpuenergy.pdf
 - relationship between cpu time and energy for training and testing respectively

[dataset]_MLP_metrics_more_training_cpuenergy_vs_testingcpuenergy.pdf
 - relationship between training energy and testing energy

pf_aggr_[dataset]_MLP_metrics[_more]_[objective1]_[objective2].csv
 - the Pareto fronts from the grid search, aggregated using median values for both objectives

pf_[dataset]_MLP_metrics[_more]_[objective1]_[objective2].csv
 - the Pareto fronts from the grid search, no aggregation

pf_[dataset]_MLP_metrics[_more]_[objective1]_[objective2].pdf
 - plot of the Pareto front from the grid search


iris-mlp-4-hist-cpu-energy.pdf
 - histogram of cpu energy measurements for iris

grid_search_results.zip
 - full set of raw data from the grid searches; includes the failed runs where energy is negative


PFs_MLP_Testing.ods
PFs_MLP_Training.ods
 - these are the spreadsheets used to show the relationships between hyperparameters and objectives for the Pareto fronts from the grid search

mop_med.zip
 - results from the NSGA-II run on diabetes, with objectives: test accuracy and training energy
 - med_fun_*.txt are the objective values from the Pareto front for each run
 - allfronts_med.csv is the above but concatenated into one file
 - med_var_*.txt are the hyperparam values from the Pareto front for each run
 - diabetes*.dat are the 1, 15, and 30th attainment surfaces

attsurface_diabetes_MLP_metrics_more_testacc_trainenergy_med.pdf
 - the figure showing the attainment surface for the mop data above

mop_mean.zip
attsurface_diabetes_MLP_metrics_more_testacc_trainenergy_mean.pdf
 - same as above, but with NSGA-II using the mean energy from repeat runs in each fitness evaluation, rather than the median
