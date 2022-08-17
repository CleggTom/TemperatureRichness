#struct to hold GLV params
struct params
    N::Int64
    r::Vector{Float64}
    a::Matrix{Float64}
end

#GLV dynamics
function dx!(dx,x,p,t)
    for i = 1:p.N
        if x[i] > 1e-6
            dx[i] = x[i] * p.r[i]
            for j = 1:p.N
                dx[i] -= x[i] * x[j] * p.a[i,j]
            end
        else
            dx[i] = 0.0
        end 
    end
end

###Assembly functions
#solve a given system to get equilibrium
function equi_solve(p,x0)
    prob = SteadyStateProblem(dx!,x0,p)
    sol = solve(prob,DynamicSS(AutoTsit5(Rosenbrock23())))
    return(sol)
end

#add a new sp to p using given generator functions
function add_Sp(p,r_func,a_func)
    r_new = vcat(p.r,r_func(1))

    a_new = a_func(p.N+1,p.N+1)
    a_new[1:p.N, 1:p.N] .= p.a
    a_new[end,end] = 1.0
    
    return(params(p.N+1, r_new, a_new))
end

#remove extict sp from p and x0
function rm_ext(p, x0)
    extant = findall(x0 .> 1e-6)
    r_new = p.r[extant]
    a_new = p.a[extant,extant]

    return(params(length(extant), r_new, a_new), x0[extant])
end

#remove the final sp from p
function rm_sp_new(p)
    r_new = p.r[1:(end-1)]
    a_new = p.a[1:(end-1),1:(end-1)]

    return(params(p.N - 1, r_new, a_new))
end

#assemble a communtiy
function assembly(r_func,a_func, N_inv = 2000)
    #start with one species
    N_vec = [1]
    C_vec = [1.0]
    p_inv = Float64[]
    r_inv = Float64[]
    a_mean = Float64[]
    r_mean = Float64[]

    #inital community
    r = r_func(1)
    a = fill(1.0, 1,1)
    p = params(1,r,a)
    x0 = equi_solve(p,[1.0]).u
    x = 1

    #assembly
    while x < N_inv
        #get prob of invasion
        #get ā
        ā = (sum(p.a) - p.N) / (p.N * (p.N - 1))
        r̄ = mean(p.r)

        p_inv_sp = mean(r_func(1000) .> mean(p.r) * ((p.N-1) * ā) / (((p.N-1) * ā) + 1))
        push!(p_inv, p_inv_sp)
        
        #get_mean_a
        push!(a_mean, ā)
        push!(r_mean, r̄)

        #new sp
        p = add_Sp(p,r_func,a_func)
        
        #check if invader can invade
        r_inv_sp = p.r[end] - sum(p.a[end,:] .* vcat(x0,0.1))
        push!(r_inv, r_inv_sp)

        if true #(r_inv_sp) > 0.0
            #simulate
            sol = equi_solve(p,vcat(x0,0.1))

            if sol.retcode == :Success #if system is stable
                #remove extinct
                p,x0 = rm_ext(p,sol.u)
                #record
                push!(N_vec,sum(x0 .> 1e-6))
            else #otherwise remove sp
                println(sol.retcode)
                p = rm_sp_new(p)
            end
        else
            # println(x, '\r')
            push!(N_vec,sum(x0 .> eps()))
            p = rm_sp_new(p)
        end

        push!(C_vec, sum(x0))

        if x % 1000 == 0
            println(x,"  ", p.N ,"   ",p_inv_sp)
        end

        x += 1 
    end

    return(N_vec,p)

end