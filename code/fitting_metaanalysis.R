# load packages
library(rTPC)
library(nls.multstart)
library(broom)
library(tidyverse)
library(progress)

#load datasets
df_meta <- read.csv("./data/database.csv") %>%
    filter(StandardisedTraitName == "Specific Growth Rate") %>%
    select(Strain = OriginalID,growth_rate = StandardisedTraitValue,Temperature = ConTemp)  %>%
    mutate(source = "meta")

df_experiments <- read.csv("./data/dataset_rates.csv") %>%
    select(Strain,growth_rate,Temperature) %>%
    mutate(source = "exp") 

#bind
df_full <- full_join(df_meta, df_experiments) %>%
    filter(growth_rate > 0.0)

#plot all data
df_full %>%
    ggplot(aes(Temperature,growth_rate))+
        geom_point()+
        facet_wrap(~source, scales = "free")

# fit two chosen model formulation in rTPC
df_nested <- df_full %>%
    nest(data = c(growth_rate,Temperature)) 

# start progress bar and estimate time it will take
number_of_models <- 1
number_of_curves <- nrow(df_nested)

# setup progress bar
pb <- progress::progress_bar$new(total = number_of_curves*number_of_models,
                                 clear = FALSE,
                                 format ="[:bar] :percent :elapsedfull")

df_nested <- df_nested %>%
    mutate(E_lm = 0.0, B0_lm = 0.0, E_ss = 0.0, B0_ss = 0.0)

for(i in 1:nrow(df_nested)){
    
    if(!is.null(pb)){
        pb$tick()
    }

    x <- df_nested$data[[i]] %>% filter(growth_rate > 0)


    
    #fit linear model
    #find peak
    i_pk <- which.max(x$growth_rate)
    #format data
    x_lin <- x %>% 
        filter(Temperature < x$Temperature[i_pk]) %>%
        mutate(T = 1/(8.617e-5) * ((1/(Temperature + 273.15)) - (1/ 286.15)))
    
    #fit
    if(nrow(x_lin) > 3){
        if(length(unique(x_lin$T)) > 3 ){
            lm_mod <- lm(log(growth_rate) ~ T, data = x_lin)
        }
    }

    if (all(tidy(lm_mod)$p.value < 0.05)) {
        df_nested$B0_lm[i] <- coef(lm_mod)[1]
        df_nested$E_lm[i] <- coef(lm_mod)[2]
    }
    
    #fit sharpe-schoolfield
    start_vals <- get_start_vals(x$Temperature, x$growth_rate, model_name = 'sharpeschoolhigh_1981')

    start_vals[is.na(start_vals)] = 0.0
    start_vals[is.infinite(start_vals)] = 0.0

    ss_mod <- tryCatch({
        nls_multstart(growth_rate ~ sharpeschoolhigh_1981(temp = Temperature, r_tref, e, eh, th, tref = 13),
                            data = x,
                            iter = c(4, 4 ,4 ,4),
                            start_lower = start_vals - 10,
                            start_upper = start_vals + 10,
                            lower = get_lower_lims(x$Temperature, x$growth_rate, model_name = 'sharpeschoolhigh_1981'),
                            upper = get_upper_lims(x$Temperature, x$growth_rate, model_name = 'sharpeschoolhigh_1981'),
                            supp_errors = 'Y')},
        error = function(e){})

    if (!is.null(ss_mod)){
        x <- tryCatch(tidy(ss_mod), error = function(x){})
        if (!is.null(x)) {
            if (!any(is.nan(x$p.value))) {
                if (all(x$p.value < 0.05)) {
                    df_nested$B0_ss[i] <- coef(ss_mod)[1]
                    df_nested$E_ss[i] <- coef(ss_mod)[2]
                }
            }
        }
    }
}

df_final <- df_nested %>%
    filter(B0_ss != 0.0, E_ss != 0.0, log(B0_ss) > -15) %>%
    mutate(B0 = log(B0_ss), E = E_ss) %>%
    group_by(source) %>%
    mutate(B0 = (B0 - mean(B0)) / sd(B0))

df_final %>%
    ggplot(aes(B0,E, color = source))+
        geom_point()

df_final %>%
    pivot_longer(c(B0, E)) %>%
    ggplot(aes(value)) +
        geom_histogram() +
        facet_wrap(name~source, scales = "free")

write_csv(df_final, "./data/summary.csv")


