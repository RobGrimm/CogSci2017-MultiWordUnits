


regress <- function(d, mwu_var, ctxtw_var, print_summaries) {
  #run regression analyses and return results
  
  # do not consider words without concreteness ratings
  d <- d[!is.na(d$concreteness),]
  
  # mode with just the covarites
  co <- lm(log(d$AoFP) ~ log(d$freq) + log(d$nsyl) + log(d$phon_n + 1) 
           + log(d$concreteness))
  # model with covariates + #MWUs
  co_MWUs <- lm(log(d$AoFP) ~ log(d$freq) + log(d$nsyl) + log(d$phon_n + 1) 
                + log(d$concreteness) + log(d[[mwu_var]] + 1))
  # model with covariates + #MWUs and #ctxt
  co_ctxtws <- lm(log(d$AoFP) ~ log(d$freq) + log(d$nsyl) + log(d$phon_n + 1) 
                  + log(d$concreteness) + log(d[[ctxtw_var]] + 1))
  
  # print regression coefficients
  if (print_summaries == TRUE) {
    print(summary(co))
    print(summary(co_MWUs))
    print(summary(co_ctxtws))
  }
  
  # R^2 for model with just the covariates
  co_fit <- summary(co)$r.squared * 100
  
  # R^2 for remaining models
  MWU_fit <- (summary(co_MWUs)$r.squared - summary(co)$r.squared) * 100
  ctxtws_fit <- (summary(co_ctxtws)$r.squared - summary(co)$r.squared) * 100
  
  # p values
  p_MWU_fit <- anova(co, co_MWUs)[["Pr(>F)"]][2]
  p_ctxtws_fit <- anova(co, co_ctxtws)[["Pr(>F)"]][2]
  
  # model with covariates + #MWUs + #ctxt
  all <- lm(log(d$AoFP) ~ log(d$freq) + log(d$nsyl) + log(d$phon_n + 1) + 
              log(d$concreteness) + log(d[[mwu_var]] + 1) + log(d[[ctxtw_var]] + 1))
  
  # print regression coefficients
  if (print_summaries == TRUE) {
    print(summary(all))
  }
  
  # R^2
  unique_MWU <- (summary(all)$r.squared - summary(co_ctxtws)$r.squared) * 100
  unique_ctxtws <- (summary(all)$r.squared - summary(co_MWUs)$r.squared) * 100
  
  # p values
  p_unique_MWU <- anova(all, co_ctxtws)[["Pr(>F)"]][2]
  p_unique_ctxtws <- anova(all, co_MWUs)[["Pr(>F)"]][2]
  
  
  return(list(covariate_fit=co_fit, 
              
              MWU_after_covariates = MWU_fit,
              p_MWU_after_covariates = p_MWU_fit,
              
              ctxtws_after_covariates = ctxtws_fit,
              p_ctxtws_after_covariates = p_ctxtws_fit,
              
              unique_MWU = unique_MWU,
              p_unique_MWU = p_unique_MWU,
              
              unique_ctxtws = unique_ctxtws,
              p_unique_ctxtws= p_unique_ctxtws))
}