# import function to run linear regressions
source('r_helper_function.R')

# read in data
be <- read.csv(paste(getwd(), sep = "/", 'BE.csv'))
na <- read.csv(paste(getwd(), sep = "/", 'NA.csv'))

################################################################################
################################################################################

# get results in variable 'r', then look at results by evaluating code snippets 
# below (e.g. get results from line 16, then evluate code snippets on 
# lines 26 -- 42 to inspect results)

# run regressions for NA corpus
r <- regress(na, mwu_var='CBL', ctxtw_var='CBL_ctxtws', print_summaries=TRUE)  
r <- regress(na, mwu_var='PBS', ctxtw_var='PBS_ctxtws', print_summaries=TRUE)  


# run regressions for BE corpus
r <- regress(be, mwu_var='CBL', ctxtw_var='CBL_ctxtws', print_summaries=TRUE)  
r <- regress(be, mwu_var='PBS', ctxtw_var='PBS_ctxtws', print_summaries=TRUE)  


r['covariate_fit']            # R^2 of model with just the covariates

r['ctxtws_after_covariates']  # increase in R^2 after adding #ctxtws to model 
                              #  with just the covariates
r['p_ctxtws_after_covariates']# p value of increase in R^2

r['MWU_after_covariates']     # increase in R^2 after adding #MWUs to model 
                              # with just he covariates
r['p_MWU_after_covariates']   # p value 

r['unique_ctxtws']            # increase in R^2 after adding #ctxtws to a model 
                              # with the covariates and #MWUs
r['p_unique_ctxtws']          # p value

r['unique_MWU']               # increase in R^2 after adding #MWUS to a model 
                              # with the covariates and #ctxtws
r['p_unique_MWU']             # p value