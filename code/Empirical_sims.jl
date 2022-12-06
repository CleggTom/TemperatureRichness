using DiffEqBase,OrdinaryDiffEq,SteadyStateDiffEq
using Distributions, LinearAlgebra, Random
using LsqFit, StatsBase
using CairoMakie, LaTeXStrings
using JLD, DelimitedFiles
using PDMats

include("./GLV/GLV.jl")

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

#confidence ellipse
function ErrorEllipse(dB, p)
    s = -2 * log(1 - p);

    λ = eigen(dB.Σ * s)
    
    t = range(0, 2 * pi, length = 100)
    
    a = λ.vectors * sqrt.(Diagonal(λ.values)) * [cos.(t)'; sin.(t)'] .+ dB.μ
    
    return(a)
end

#empirical data plot
df_all = readdlm("./data/summary.csv", ',')
#split datasets
meta_ind = findall(df_all[:,2] .== "meta")
exp_ind = findall(df_all[:,2] .== "exp")

col_ind = findall(df_all[1,:] .∈ Ref(["B0","E"]))

#meta
df_meta = float.(df_all[meta_ind,col_ind])
#fit MVNormal
dB_meta = Distributions.fit_mle(MvNormal{}, df_meta')

#Experiments
df_exp = float.(df_all[exp_ind,col_ind])
#fit MVNormal
dB_exp = Distributions.fit_mle(MvNormal, df_exp')

#predicting
N_rep = 100
prob = 0.5

begin
    #predicting
    T_vec = -2.0:0.2: 2.0
    T_plot = GLV.ΔT_to_C.(T_vec, 13)

    #get exp params
    uE = abs(dB_exp.μ[2])
    vB0,c,c,vE = dB_exp.Σ[:]

    r = Dict(:uB0 => 0.0, :vB0 =>  vB0, :uE => uE, :vE => vE, :Σ => c)
    a = Dict(:uB0 => -5.0, :vB0 => vB0, :uE => uE, :vE => vE, :Σ => c)

    sim_exp,pred_exp = pred_richness(r,a,T_vec,N_rep,prob)

    #get meta params
    uE = dB_meta.μ[2]
    vB0,c,c,vE = dB_meta.Σ[:]

    r = Dict(:uB0 => 0.0, :vB0 =>  vB0, :uE => uE, :vE => vE, :Σ => c)
    a = Dict(:uB0 => -5.0, :vB0 => vB0, :uE => uE, :vE => vE, :Σ => c)

    sim_meta,pred_meta = pred_richness(r,a,T_vec,N_rep,prob)
end

function boltz(B0,E,ΔT)
    B0  -E * ΔT
end

N_rep = 1000

begin
    #set up grid
    begin
        #plotting
        f = Figure(resolution = (1000, 500))

        ga1 = f[1,1][1,1] = GridLayout()
        ga2 = f[1,1][1,2] = GridLayout()

        gb1 = f[2, 1][1,1] = GridLayout()
        gb2 = f[2, 1][1,2] = GridLayout()

        # gc = f[1:2, 2] = GridLayout()

        # #set a xes
        ag1 = ga1[1,1] = Axis(f, height = 25)
        ag2 = ga1[2,1] = Axis(f, xlabel = "Log(B0)", ylabel = "E")
        ag3 = ga1[2,2] = Axis(f, width = 25)
        ag4 = ga2[1,1] = Axis(f, xlabel = "Temperature", ylabel = "Log(Growth rate)")    

        #set axes
        am1 = gb1[1,1] = Axis(f, height = 25)
        am2 = gb1[2,1] = Axis(f, xlabel = "Log(B0)", ylabel = "E")
        am3 = gb1[2,2] = Axis(f, width = 25)
        am4 = gb2[1,1] = Axis(f, xlabel = "Temperature", ylabel = "Log(Growth rate)")    
        
        a_richness = gc = Axis(f[1:2, 2], xlabel = "Temperature", ylabel = "Richness", width = 400)

        colgap!(ga1,5)
        rowgap!(ga1,5)

        colgap!(gb1,5)
        rowgap!(gb1,5)

        # #Labels
        Label(f[1,1][1,1,TopLeft()],"A")
        Label(f[1,1][1,2,TopLeft()],"B")
        Label(f[2,1][1,1,TopLeft()],"C")
        Label(f[2,1][1,2,TopLeft()],"D")
        Label(f[1:2,2,TopLeft()],"E")

        #scatter plot
        linkxaxes!(am2, am1, ag2, ag1)
        linkyaxes!(am2, am3, ag2, ag3)
        linkyaxes!(ag4, am4)
        
    end

    #calculate conf ellipse
    ellipse_exp = ErrorEllipse(dB_exp, 0.95)
    ellipse_meta = ErrorEllipse(dB_meta, 0.95)
    
    begin
        #exp
        Makie.hist!(ag1, df_exp[:,1])
        # plot
        Makie.scatter!(ag2, float.(df_exp[:,1]), float.(df_exp[:,2]), 
            color = ("cornflowerblue",0.35))

        Makie.lines!(ag2, ellipse_exp[1,:], ellipse_exp[2,:], color = "black")
        Makie.hist!(ag3, df_exp[:,2], direction = :x)
        hidedecorations!(ag1, grid = false)
        hidedecorations!(ag3, grid = false)

        #plot TPC curves
        [lines!(ag4, T_plot, boltz.(df_exp[i,1],df_exp[i,2],T_vec), color = ("black", 0.1)) for i = 1:size(df_exp)[1]]
        
        #plot distributions
        sim_exp_data = rand(dB_exp,N_rep)'
        for T = range(-2,2,length = 12)
            boxplot!(ag4, repeat([GLV.ΔT_to_C(T, 13.0)], N_rep), boltz.(sim_exp_data[:,1], sim_exp_data[:,2],T) , color = "black", show_outliers = false)
        end

        #get min variance temp
        min_var_exp = [GLV.ΔT_to_C(dB_exp.Σ[1,2] / dB_exp.Σ[2,2], 13.0)]
        vlines!(ag4,min_var_exp, ymax = [0.7], color = "black", linestyle = "-")
    end

    begin
        #meta
        Makie.hist!(am1, df_meta[:,1])
        # plot
        Makie.scatter!(am2, float.(df_meta[:,1]), float.(df_meta[:,2]), 
            color = ("crimson",0.35))
    
        Makie.lines!(am2, ellipse_meta[1,:], ellipse_meta[2,:], color = "black")
        Makie.hist!(am3, df_meta[:,2], direction = :x)
    
        hidedecorations!(am1, grid = false)
        hidedecorations!(am3, grid = false)

        #plot TPC curves
        [lines!(am4, T_plot, boltz.(df_meta[i,1],df_meta[i,2],T_vec), color = ("black", 0.05)) for i = 1:size(df_meta)[1]]
        #get trait Distributions
        sim_meta_data = rand(dB_meta,N_rep)'
        for T = range(-2,2,length = 12)
            boxplot!(am4, repeat([GLV.ΔT_to_C(T, 13.0)], N_rep), boltz.(sim_meta_data[:,1], sim_meta_data[:,2],T) , color = "black", show_outliers = false)
        end
        #get min variance temp
        min_var_meta = [GLV.ΔT_to_C(dB_meta.Σ[1,2] / dB_meta.Σ[2,2], 13.0)]
        vlines!(am4,min_var_meta, ymax = [0.7], color = "black", linestyle = "-")
    end

    begin
        Makie.scatter!(a_richness, T_plot, sim_exp, color = "cornflowerblue", label = "Experiments")
        Makie.lines!(a_richness, T_plot, pred_exp, color = "cornflowerblue")

        Makie.scatter!(a_richness, T_plot, sim_meta, color = "crimson", label = "Meta-analysis")
        Makie.lines!(a_richness, T_plot, pred_meta , color = "crimson")

        ylims!(a_richness, (0, 1.1 * maximum(hcat(pred_exp,pred_meta))))
        Legend(f[1:2, 2], a_richness, orientation = :horizontal, halign = :center, valign = :bottom, framevisible = false)
    end

    save("./docs/Figures/grw_data.pdf", f)

    f
end


