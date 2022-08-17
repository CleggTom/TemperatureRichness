### Prediction functions
#predicting based on traits
function interaction_bound(a,N)
    (a*(N-1)) / (a*(N-1) + 1)
end

#Prob of feas as a function of ā and the CDF of κ
function P_feas(a,cdf_k,N)
    bound = interaction_bound(a,N)
    return( (1 - cdf_k(bound))^N )
end

#function to calculate the richness from a given lognormal distribution of a and k
#works at probabilty threshold θ
function richness(θ,da,dK)
    #normalise k
    dk = LogNormal(-(dK.σ^2) /2 , dK.σ)
    
    for N = 1:0.05:500
        bound = interaction_bound(mean(da),N)
        if (1 - cdf(dk,bound))^N < θ
            # println( (1-cdf(dk,bound))^N ) 
            return(N)
        end
    end
    return(500)
end

function richness_empirical(θ,a,K)        
    for N = 1:0.05:300
        bound = interaction_bound(mean(a),N)
        cdf = StatsBase.ecdf(K ./ mean(K))

        if (1 - cdf(bound))^N < θ
            # println( (1-cdf(dk,bound))^N ) 
            return(N)
        end
    end
    return(300)
end