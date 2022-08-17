### Temperature
#convert dict of bivarate normal into a single distribution
function trait_temp(p,ΔT)
    @assert isposdef([p[:vB0] p[:Σ] ; p[:Σ] p[:vE]]) "Covariance matrix is not positive definate"
    return(LogNormal(p[:uB0] - ΔT*p[:uE], sqrt(p[:vB0] + (p[:vE] * ΔT^2) - 2*p[:Σ]*ΔT)))
end

#sample traits from a given B0-E distribution at temperature ΔT
function temp_sampler(ΔT, p, x...)
    return( (rand(trait_temp(p,ΔT),x...)) )
end

#function to convert ΔT units into celcius.
function ΔT_to_C(ΔT,T_ref)
    (1 / (8.617e-5*ΔT + (1 / (T_ref+273.15)))) - 273.15
end
