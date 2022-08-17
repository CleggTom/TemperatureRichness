#GLV assembly simulations

#contains code to run assemlby simulations at different temperatures. 

using Revise
using Random, Distributions
using CairoMakie, JLD
using Colors

include("./GLV/GLV.jl")

### Test @1 temp
begin
    Random.seed!(1)
    dr = LogNormal(0.0,1.0)
    da = LogNormal(-4,1.0)
    #test Distribution functions
    function random_r(x...)
        rand(dr,x...)
    end

    function random_a(x...)
        rand(da,x...)
    end

    #κ cdf
    cdf_Knorm(x) = cdf(LogNormal(dr.σ^2  / 2, dr.σ),x)

    N_inv = 1000
    N_vec,p = GLV.assembly(random_r,random_a, N_inv)
end

#plot single assembly
begin
    fig = Figure()
    ax1 = Axis(fig[1,1], ylabel = "Richness", show_axis = false, xtickalign = 1)
    CairoMakie.lines!(ax1, N_vec)
   
    fig
end

### Temperature example
begin
    #define TPC params
    r = Dict(:uB0 => 0.0, :vB0 => 0.2, :uE => 0.2, :vE => 0.1, :Σ => 0.0)
    a = Dict(:uB0 => -2.0, :vB0 => 0.2, :uE => 0.2, :vE => 0.1, :Σ => 0.0)
    #temperature vector
    T_vec = -1.5:0.3:1.5
    
    #results
    res = []
    nrep = 10
    N_inv = 1000
    for (i,T) = enumerate(T_vec)
        print(i,"\r")
        f_r(x...) = rand(GLV.trait_temp(r, T), x...)
        f_a(x...) = rand(GLV.trait_temp(a, T), x...)

        push!(res,[GLV.assembly(f_r,f_a,N_inv) for i = 1:nrep])
    end

    #plot timeseries
    f = Figure()
    ax1 = Axis(f[1,1])
    ax2 = Axis(f[1,2])

    c_vec = range(colorant"red",colorant"blue", length = length(T_vec))


    [[lines!(ax1, x[1], color = (c_vec[i], 0.2)) for x = res[i]] for i = eachindex(T_vec)]

    #fit asymptotes, final richness
    N_end = hcat([[x[i][1][end] for i = 1:nrep] for x = res]...)
    N_pred = [GLV.richness(1e-6 ,GLV.trait_temp.([a,r],T)...) for T = T_vec]

    [scatter!(ax2, T_vec, N_end[i,:], color = "black") for i = 1:nrep ]
    lines!(ax2, T_vec, N_pred, color = "red")

    f
end

###Figures
#load "real" parameters
a = load("./data/smith_mvnorm.jld")["dB_grw"]
dR = Distributions.MvNormal(a.μ, a.Σ.mat)

#variation
μ_vec = [0.0,0.6]
σ_vec = [0.01,0.2]
Σ_vec = [-0.1,0.0]

#temp vec
T_vec = -1.5:0.2:1.5
nrep = 3
N_inv = 2000


#varying average thermal sensitvtiy
begin
    Random.seed!(1)
    #define parameter ranges
    r = [Dict(:uB0 => 0.0, :vB0 => 0.2, :uE => x, :vE => 0.01, :Σ => 0.0) for (i,x) = enumerate(μ_vec)]
    a = [Dict(:uB0 => -2.0, :vB0 => 0.2, :uE => x, :vE => 0.01, :Σ => 0.0) for (i,x) = enumerate(μ_vec)]

    #results
    N_sim_μ = Array{Float64, 3}(undef,length(μ_vec),length(T_vec),nrep)
    N_pred_μ = Array{Float64, 2}(undef,length(μ_vec),length(T_vec))

    for μ = eachindex(μ_vec)
        res = []
        for (i,T) = enumerate(T_vec)
            println(μ,"  ",i, " of ", length(T_vec),"\r")
            #define trait samplers
            f_r(x...) = GLV.temp_sampler(T, r[μ], x...)
            f_a(x...) = GLV.temp_sampler(T, a[μ], x...)
            
            for j = 1:nrep
                sim = GLV.assembly(f_r,f_a,N_inv)
                #save richness
                N_sim_μ[μ,i,j] = sim[1][end]
            end

            N_pred_μ[μ,i] = GLV.richness(1e-6 ,GLV.trait_temp.([a[μ],r[μ]],T)...)

        end
    end
end

#varying variation
begin
    Random.seed!(1)
    r = [Dict(:uB0 => 0.0, :vB0 => 0.2, :uE => 0.1, :vE => x, :Σ => 0.0) for (i,x) = enumerate(σ_vec)]
    a = [Dict(:uB0 => -2.0, :vB0 => 0.2, :uE => 0.1, :vE => x, :Σ => 0.0) for (i,x) = enumerate(σ_vec)]

        #results
        N_sim_σ = Array{Float64, 3}(undef,length(σ_vec),length(T_vec),nrep)
        N_pred_σ = Array{Float64, 2}(undef,length(σ_vec),length(T_vec))

        for σ = eachindex(σ_vec)
            res = []
            for (i,T) = enumerate(T_vec)
                println(σ,"  ",i, " of ", length(T_vec),"\r")
                #define trait samplers
                f_r(x...) = GLV.temp_sampler(T, r[σ], x...)
                f_a(x...) = GLV.temp_sampler(T, a[σ], x...)
                
                for j = 1:nrep
                    sim = GLV.assembly(f_r,f_a,N_inv)
                    #save richness
                    N_sim_σ[σ,i,j] = sim[1][end]
                end

                N_pred_σ[σ,i] = GLV.richness(1e-6 ,GLV.trait_temp.([a[σ],r[σ]],T)...)

            end
        end
end

#varying Covariance in thermal sensitvtiy and B0
begin
    Random.seed!(1)
    r = [Dict(:uB0 => 0.0, :vB0 => 0.2, :uE => 0.1, :vE => 0.1, :Σ => x) for (i,x) = enumerate(Σ_vec)]
    a = [Dict(:uB0 => -2.0, :vB0 => 0.2, :uE => 0.1, :vE => 0.1, :Σ => x) for (i,x) = enumerate(Σ_vec)]

    #results
    N_sim_Σ = Array{Float64, 3}(undef,length(Σ_vec),length(T_vec),nrep)
    N_pred_Σ = Array{Float64, 2}(undef,length(Σ_vec),length(T_vec))

    for Σ = eachindex(Σ_vec)
        res = []
        for (i,T) = enumerate(T_vec)
            println(Σ,"  ",i, " of ", length(T_vec),"\r")
            #define trait samplers
            f_r(x...) = GLV.temp_sampler(T, r[Σ], x...)
            f_a(x...) = GLV.temp_sampler(T, a[Σ], x...)
            
            for j = 1:nrep
                sim = GLV.assembly(f_r,f_a,N_inv)
                #save richness
                N_sim_Σ[Σ,i,j] = sim[1][end]
            end

            N_pred_Σ[Σ,i] = GLV.richness(1e-6 ,GLV.trait_temp.([a[Σ],r[Σ]],T)...)

        end
    end
end


#plotting with predictions
begin
    T_plot = GLV.ΔT_to_C.(T_vec,13)

    f = Makie.Figure(resolution = (550, 750))

    a11 = f[1,1][1,1] = Axis(f)
    a12 = f[1,1][1,2] = Axis(f)

    a21 = f[2,1][1,1] = Axis(f, ylabel = "Richness N")
    a22 = f[2,1][1,2] = Axis(f)

    a31 = f[3,1][1,1] = Axis(f)
    a32 = f[3,1][1,2] = Axis(f)

    linkyaxes!(a11,a12)
    linkyaxes!(a21,a22)
    linkyaxes!(a31,a32)



    Label(f[1,1][1,1,TopLeft()], "A",padding = (0.0,0.0,5,0.0))
    Label(f[1,1][1,2,TopLeft()], "B",padding = (0.0,0.0,5,0.0))
    Label(f[2,1][1,1,TopLeft()], "C",padding = (0.0,0.0,5,0.0))
    Label(f[2,1][1,2,TopLeft()], "D",padding = (0.0,0.0,5,0.0))
    Label(f[3,1][1,1,TopLeft()], "E",padding = (0.0,0.0,5,0.0))
    Label(f[3,1][1,2,TopLeft()], "F",padding = (0.0,0.0,5,0.0))

    Label(f[3,1, Bottom()], "Temperature °C", padding = (0.0,0,0,30))

    #plotting mean plot
    c = ("black", 0.6)
    [Makie.scatter!(a11, T_plot, N_sim_μ[1,:,i], color = c) for i = 1:nrep]
    Makie.lines!(a11, T_plot, N_pred_μ[1,:], color = "red")

    [Makie.scatter!(a12, T_plot, N_sim_μ[2,:,i], color = c) for i = 1:nrep]
    Makie.lines!(a12, T_plot, N_pred_μ[2,:], color = "red")

    #plotting var plots
    [Makie.scatter!(a21, T_plot, N_sim_σ[1,:,i], color = c) for i = 1:nrep]
    Makie.lines!(a21, T_plot, N_pred_σ[1,:], color = "red")

    [Makie.scatter!(a22, T_plot, N_sim_σ[2,:,i], color = c) for i = 1:nrep]
    Makie.lines!(a22, T_plot, N_pred_σ[2,:], color = "red")

    #plotting var plots
    [Makie.scatter!(a31, T_plot, N_sim_Σ[1,:,i], color = c) for i = 1:nrep]
    Makie.lines!(a31, T_plot, N_pred_Σ[1,:], color = "red")

    [Makie.scatter!(a32, T_plot, N_sim_Σ[2,:,i], color = c) for i = 1:nrep]
    Makie.lines!(a32, T_plot, N_pred_Σ[2,:], color = "red")

        
    f
end

save("./docs/Figures/assembly.pdf", f)
