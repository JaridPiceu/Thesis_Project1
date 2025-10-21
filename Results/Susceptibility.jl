using Plots
using CSV, DataFrames
using TNRKit

include("../TensorMaker/TensorFactory.jl")


#Reproduce results from the paper of Kadoh: fig 2, 3 and 4

#NOTE: in fig 2, we use absolute of different <ϕ>_h and <ϕ>_0 instead of difference because of plotting Logarithm

# PARAMETERS
K = 16
niter=15
ndimtrunc = 16 


function getΦ4ExpValue(K, μ0, λ, h ;niter=niter, ndimtrunc=ndimtrunc)
    Tpure = getTensor(K, μ0, λ, h)
    Timp1 = getImp1Tensor(K, μ0, λ, h)
    Timp2 = getImp2Tensor(K, μ0, λ, h)

    scheme = Imp(Tpure, Timp1, Timp1, Timp2)
    _, _ = run!(scheme, truncdim(ndimtrunc), maxiter(niter))

    norm_pure = norm(@tensor scheme.T[1 2; 2 1])
    norm_imp1 = norm(@tensor scheme.T_imp_order1_1[1 2; 2 1])
    return norm_imp1 / norm_pure
end


# Fig 2

function getDataFig2(; niter=niter, ndimtrunc = ndimtrunc, K = K)
    # Parameters used
    λ = 0.05
    μ0 = -0.1006174
    hs = [1e-6, 1e-8, 1e-10, 1e-12]
    # In paper: K = 256
    # In paper: ndimtrunc = 32

    results = DataFrame(h=Float64[], L=Int[], expϕ=Float64[])

    for h in hs
        Tpure = getTensor(K, μ0, λ, h)
        Timp1 = getImp1Tensor(K, μ0, λ, h)
        Timp2 = getImp2Tensor(K, μ0, λ, h)

        scheme = ImpurityHOTRG(Tpure, Timp1, Timp1, Timp2)

        # Calculate <phi> step by step
        for i in 1:niter
            TNRKit.step!(scheme, truncdim(ndimtrunc))
            TNRKit.finalize!(scheme)

            norm_pure = norm(@tensor scheme.T[1 2; 2 1])
            norm_imp1 = norm(@tensor scheme.T_imp_order1_1[1 2; 2 1])
        
            ans = abs((norm_imp1/norm_pure)/h)
            push!(results, (h=h, L=2^i, expϕ=ans))
        end
    end

    # Ensure Data directory exists
    if !isdir("Results/Data")
        mkpath("Results/Data")
    end
    
    filepath = joinpath("Results/Data", "Fig2.csv")
    CSV.write(filepath, results)
end


function plotDataFig2(; filename=joinpath("Results/Data", "Fig2.csv"))
    df = CSV.read(filename, DataFrame)
    plt = plot(xscale=:log10, yscale=:log10, xlabel="System size L", ylabel="<ϕ>/h", title="Susceptibility", legend=:topleft)

    for h in unique(df.h)
        subset = filter(row -> row.h == h, df)
        scatter!(plt, subset.L, subset.expϕ, label="h=$(h)")
    end

    pathplot = joinpath("Results/Plots", "Fig2.png")
    savefig(plt, pathplot)
    return plt
end

getDataFig2()
plotDataFig2()



# FIG 3

# PARAMETERS
npoints = 5

function getDataFig3(; λ=0.05, μ0=-0.1006174, K=K, ndimtrunc=ndimtrunc, niter=niter, npoints=npoints)
    # Define h range
    hs = 10 .^ range(-12, -2; length=npoints)
    expϕs = Float64[]

    # Compute ⟨ϕ⟩/h for each h
    for h in hs
        expϕ = getΦ4ExpValue(K, μ0, λ, h; niter=niter, ndimtrunc=ndimtrunc)
        push!(expϕs, expϕ / h)
    end

    # Ensure Data directory exists
    if !isdir("Results/Data")
        mkpath("Results/Data")
    end

    df = DataFrame(h=hs, expϕ=expϕs)

    filepath = joinpath("Results/Data", "Fig3.csv")
    CSV.write(filepath, df)
end



function getPlotFig3(; filepath=joinpath("Results/Data", "Fig3.csv"))
    # Load data
    df = CSV.read(filepath, DataFrame)

    # Plot
    plt = scatter(
        df.h, df.expϕ,
        xscale = :log10,
        yscale = :log10,
        xlabel = "h",
        ylabel = "⟨ϕ⟩ / h",
        title = "Susceptibility",
        legend = false,
        markersize = 5
    )

    # Ensure Data directory exists
    if !isdir("Results/Plots")
        mkpath("Results/Plots")
    end
    
    plotpath = joinpath("Results/Plots", "Fig3.png")

    savefig(plt, plotpath)
end

getDataFig3()
getPlotFig3()


# FIG 4

function getDataFig4(; λ=0.05, μ0=-0.1006174, K=K, ndimtrunc=ndimtrunc, niter=niter, npoints=npoints)
    # Define h range
    hs = 10 .^ range(-12, -11; length=npoints)
    expϕs = Float64[]

    # Compute ⟨ϕ⟩/h for each h
    for h in hs
        expϕ = getΦ4ExpValue(K, μ0, λ, h; niter=niter, ndimtrunc=ndimtrunc)
        push!(expϕs, expϕ / h)
    end

    # Ensure Data directory exists
    if !isdir("Results/Data")
        mkpath("Results/Data")
    end

    df = DataFrame(h=hs, expϕ=expϕs)

    filepath = joinpath("Results/Data", "Fig4.csv")
    CSV.write(filepath, df)
end


function getPlotFig4(; filepath=joinpath("Results/Data", "Fig4.csv"))
    # Load data
    df = CSV.read(filepath, DataFrame)

    # Plot
    plt = scatter(
        df.h, df.expϕ,
        xlabel = "h",
        ylabel = "⟨ϕ⟩ / h",
        title = "Susceptibility",
        legend = false,
        markersize = 5
    )

    # Ensure Data directory exists
    if !isdir("Results/Plots")
        mkpath("Results/Plots")
    end
    
    plotpath = joinpath("Results/Plots", "Fig4.png")

    savefig(plt, plotpath)
end

getDataFig4()
getPlotFig4()


