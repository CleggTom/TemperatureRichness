### Asymptote fitting
#asymtote functions
function asymptote(x,p)
    p[1] .* (x) ./ (x .+ p[2])
end

#gives clean array output
function clean_assembly(N_vec)
    hcat([[x for x = i] for i = zip(N_vec...)]...)'
end

#fits asymtote
function fit_asymptote(N_vec, inv_max)
    clean_data = clean_assembly(N_vec)
    i = min(inv_max, size(clean_data)[1])

    clean_data = clean_data[1:i, : ]
    ydata = clean_data[:]
    xdata = repeat(1:size(clean_data)[1],size(clean_data)[2])

    p0 = [maximum(clean_data), 10.5]
    fit = curve_fit(asymptote,xdata,ydata,p0)

    return(fit.param)
end

function get_N_end(N_vec)
    map(x -> x[end], N_vec)
end