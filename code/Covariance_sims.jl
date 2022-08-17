using DiffEqBase,OrdinaryDiffEq,SteadyStateDiffEq
using Distributions, LinearAlgebra, Random
using LsqFit, StatsBase
using CairoMakie, LaTeXStrings
using JLD, DelimitedFiles

include("./GLV/GLV.jl")

### Test @1 temp
begin
    Random.seed!(1)
    dr = LogNormal(0.0,0.447)
    da = LogNormal(-3.0,0.447)

    #test Distribution functions
    function random_r(x...)
        rand(dr,x...)
    end

    function random_a(x...)
        rand(da,x...)
    end

    #κ cdf
    cdf_Knorm(x) = cdf(LogNormal(dr.σ^2  / 2, dr.σ),x)

    #generate systems of N sizer
    N_max = ceil(GLV.richness(1e-2,da,dr))
    N_vec = Int.(1:2:(N_max*3))

    rep = 20
    res = zeros(rep,length(N_vec))

    for i = 1:rep
        for (j,N) = enumerate(N_vec)
            #make system
            r = rand(dr, N)
            a = rand(da, N, N)
            [a[k,k] = 1.0 for k = 1:N]

            p = GLV.params(N, r, a)

            sol = GLV.equi_solve(p,ones(N))

            res[i,j] = sum(sol.u .> 1e-6) == N

        end
    end

    #predictions
    p_vec = 10 .^ (-6:0.1:0.0)
    N_vec_pred = GLV.richness.(p_vec, Ref(da),Ref(dr))

    #plot
    f = Figure()
    ax = Axis(f[1,1], xlabel = "Richness", ylabel = "log10(p_feas)")
    CairoMakie.lines!(ax, N_vec,log10.(mean(res,dims = 1)[:]), label = "simualted")
    CairoMakie.lines!(ax, N_vec_pred, log10.(p_vec), label = "predicted")
    axislegend(ax)
    f
end

#predcit richness
function pred_richness(r,a,T_vec,N_rep,prob)
    simulated_richness = Array{Float64}(undef, length(T_vec))
    predicted_richness = zeros(length(T_vec))

    for (i,T) = enumerate(T_vec)
        #get trait Distributions
        dr,da = GLV.trait_temp(r, T), GLV.trait_temp(a, T)
        
        #predicted
        predicted_richness[i] = GLV.richness(prob, da, dr)
        N = Int(ceil(predicted_richness[i]))
        
        N_vec = max(N-10,1):(N+10)

        #generators
        f_r(x...) = rand(dr, x...)
        f_a(x...) = rand(da, x...)
        
        #loop over Nvals
        p_feas = zeros(length(N_vec))
        for (k,N) = enumerate(N_vec)
            feas_vec = zeros(N_rep)
            for j = 1:N_rep
                #generate params
                r_ = f_r(N)
                a_ = f_a(N,N)
                [a_[ii,ii] = 1.0 for ii = 1:N]
                #make params
                p = GLV.params(N, r_, a_)
                #solve
                sol = GLV.equi_solve(p,ones(N))

                feas_vec[j] = sum(sol.u .> 1e-6) == N
            end
            p_feas[k] = mean(feas_vec)
        end
        #get bounds on richness
        N_indx = findmin(abs.(p_feas .- 0.5))
        simulated_richness[i] = N_vec[N_indx[2]]
    end
    return(simulated_richness, predicted_richness)
end

#test predictions
begin
    r = Dict(:uB0 => 0.0, :vB0 => 0.2, :uE => 0.1, :vE => 0.1, :Σ => 0.0)
    a = Dict(:uB0 => -3.0, :vB0 => 0.2, :uE => 0.1, :vE => 0.1, :Σ => 0.0)

    sim,pred = pred_richness(r,a,T_vec,50,0.5)

    f = Figure()
    ax = Axis(f[1,1])
    Makie.scatter!(ax, T_vec, sim)
    Makie.lines!(ax, T_vec, pred)
    f
end

#simulate over different parameter combinations
N_rep = 50
prob = 0.5

#mean
begin
    r = Dict(:uB0 => 0.0, :vB0 => 0.2, :uE => 0.2, :vE => 0.01, :Σ => 0.0)
    a = Dict(:uB0 => -3.5, :vB0 => 0.2, :uE => 0.2, :vE => 0.01, :Σ => 0.0)

    #mean
    sim_μ_1,pred_μ_1 = pred_richness(r,a,T_vec,N_rep,prob)
    #alter μE
    r[:uE] = 0.6
    a[:uE] = 0.6
    sim_μ_2,pred_μ_2 = pred_richness(r,a,T_vec,N_rep,prob)

    f = Figure()
    ax = Axis(f[1,1])
    Makie.scatter!(ax, T_vec, sim_μ_1)
    Makie.lines!(ax, T_vec, pred_μ_1)

    Makie.scatter!(ax, T_vec, sim_μ_2)
    Makie.lines!(ax, T_vec, pred_μ_2)
    f
end

#var
begin
    r = Dict(:uB0 => 0.0, :vB0 => 0.2, :uE => 0.2, :vE => 0.01, :Σ => 0.0)
    a = Dict(:uB0 => -3.5, :vB0 => 0.2, :uE => 0.2, :vE => 0.01, :Σ => 0.0)

    #mean
    sim_σ_1,pred_σ_1 = pred_richness(r,a,T_vec,N_rep,prob)
    #alter μE
    r[:vE] = 0.2
    a[:vE] = 0.2
    sim_σ_2,pred_σ_2 = pred_richness(r,a,T_vec,N_rep,prob)

    f = Figure()
    ax = Axis(f[1,1])
    Makie.scatter!(ax, T_vec, sim_σ_1)
    Makie.lines!(ax, T_vec, pred_σ_1)

    Makie.scatter!(ax, T_vec, sim_σ_2)
    Makie.lines!(ax, T_vec, pred_σ_2)
    f
end

#cov
begin
    r = Dict(:uB0 => 0.0, :vB0 => 0.2, :uE => 0.2, :vE => 0.2, :Σ => 0.0)
    a = Dict(:uB0 => -3.5, :vB0 => 0.2, :uE => 0.2, :vE => 0.2, :Σ => 0.0)

    #mean
    sim_Σ_1,pred_Σ_1 = pred_richness(r,a,T_vec,N_rep,prob)
    #alter μE
    r[:Σ] = -0.1
    a[:Σ] = -0.1
    sim_Σ_2,pred_Σ_2 = pred_richness(r,a,T_vec,N_rep,prob)

    f = Figure()
    ax = Axis(f[1,1])
    Makie.scatter!(ax, T_vec, sim_Σ_1)
    Makie.lines!(ax, T_vec, pred_Σ_1)

    Makie.scatter!(ax, T_vec, sim_Σ_2)
    Makie.lines!(ax, T_vec, pred_Σ_2)
    f
end

#full plot
begin
    f = Figure(resolution = (900,300))
    ax = [Axis(f[1,i]) for i = 1:3]

    # Makie.xlabel!(, "Temperature")

    linkyaxes!(ax[1],ax[2],ax[3])

    T_plot = GLV.ΔT_to_C.(T_vec,13.0)

    #mean
    Makie.scatter!(ax[1], T_plot, sim_μ_1, color = "darkblue", label = "0.1")
    Makie.lines!(ax[1], T_plot, pred_μ_1, color = "darkblue")
    Makie.scatter!(ax[1], T_plot, sim_μ_2, color = "crimson", label = "0.6")
    Makie.lines!(ax[1], T_plot, pred_μ_2, color = "crimson")
    # axislegend(ax[1], name = "a")
    
    #var
    Makie.scatter!(ax[2], T_plot, sim_σ_1, color = "darkblue", label = "0.01")
    Makie.lines!(ax[2], T_plot, pred_σ_1, color = "darkblue")
    Makie.scatter!(ax[2], T_plot, sim_σ_2, color = "crimson", label = "0.2")
    Makie.lines!(ax[2], T_plot, pred_σ_2, color = "crimson")
    #cov
    Makie.scatter!(ax[3], T_plot, sim_Σ_1, color = "darkblue", label = "0.0")
    Makie.lines!(ax[3], T_plot, pred_Σ_1, color = "darkblue")
    Makie.scatter!(ax[3], T_plot, sim_Σ_2, color = "crimson", label = "-0.1")
    Makie.lines!(ax[3], T_plot, pred_Σ_2, color = "crimson")

    f[1,1] = Legend(f, ax[1], "mean", tellwidth = false, halign = :right, valign = :top)
    f[1,2] = Legend(f, ax[2], "variance", tellwidth = false, halign = :right, valign = :top)
    f[1,3] = Legend(f, ax[3], "covariance", tellwidth = false,  halign = :right, valign = :top)

    Label(f[1,1:3,Bottom()], "Temperature °C", padding = (0,0,0,35))
    Label(f[1,1,Left()], "Richness N", padding = (0,35,0,10), rotation = 3.14 / 2)
    
    Label(f[1,1,TopLeft()], "A",padding = (0.0,0.0,5,0.0))
    Label(f[1,2,TopLeft()], "B",padding = (0.0,0.0,5,0.0))
    Label(f[1,3,TopLeft()], "C",padding = (0.0,0.0,5,0.0))


    save("./docs/Figures/LV_sims.pdf", f)

    f
end

