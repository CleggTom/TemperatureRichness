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
    mutate(E = 0.0, B0 = 0.0)

for(i in 1:nrow(df_nested)){
    
    if(!is.null(pb)){
        pb$tick()
    }

    x <- df_nested$data[[i]]

    #find peak
    i_pk <- which.max(x$growth_rate)

    x <- x %>% 
    filter(Temperature < x$Temperature[i_pk],growth_rate > 0 ) %>%
    mutate(T = 1/(8.617e-5) * ((1/(Temperature + 273.15)) - (1/ 286.15)))
    
    if(nrow(x) > 3){
        if(length(unique(x$T)) > 3 ){
    model <- lm(log(growth_rate) ~ T, data = x)
        }
    }

    if(all(tidy(model)$p.value < 0.05)){
        df_nested$B0[i] <- coef(model)[1] 
        df_nested$E[i] <- -coef(model)[2]  
    }
}

df_nested <- df_nested %>% 
    filter(B0 != 0.0, B0 > -30, E < 2) 

df_nested %>%
    ggplot(aes(B0,E, color = source))+
        geom_point()

write_csv(df_nested, "./data/summary.csv")


