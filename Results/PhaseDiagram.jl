using Plots
using CSV, DataFrames
using TNRKit
using TensorKit

include("../TensorMaker/TensorFactory.jl")

# PARAMETERS
niter = 15
ndimtrunc = 8
K = 8
μ0_range = range(-10, 10, length = 6)
λ_range = range(0.1, 1, length = 3)


"""
Get ⟨ϕ⁴⟩ expectation value using impurity tensors. The algorithm used is Impurity HOTRG.
"""
function getΦ4ExpValueSquared(K, μ0, λ, h; niter = 15, ndimtrunc = 16)
    Tpure = getTensor(K, μ0, λ, h)
    Timp1 = getImp1Tensor(K, μ0, λ, h)
    Timp2 = getImp2Tensor(K, μ0, λ, h)

    scheme = ImpurityHOTRG(Tpure, Timp1, Timp1, Timp2)
    _, _ = run!(scheme, truncdim(ndimtrunc), maxiter(niter))

    norm_pure = norm(@tensor scheme.T[1 2; 2 1])
    norm_imp2 = norm(@tensor scheme.T_imp_order2[1 2; 2 1])
    return norm_imp2 / norm_pure
end


function getSSB_Data(;
        K = K, h = 0, niter = niter, ndimtrunc = ndimtrunc,
        μ0_range = μ0_range,
        λ_range = λ_range
    )

    # Ensure Data directory exists
    if !isdir("Results/Data")
        mkpath("Results/Data")
    end

    results = DataFrame(μ0 = Float64[], λ = Float64[], value = Float64[])

    for μ0 in μ0_range, λ in λ_range
        val = getΦ4ExpValueSquared(K, μ0, λ, h; niter = niter, ndimtrunc = ndimtrunc)
        push!(results, (μ0, λ, val))
        println("Computed μ0 = $μ0, λ = $λ → value = $val")
    end

    filepath = joinpath("Results/Data", "PhaseDiagram.csv")
    CSV.write(filepath, results)
    return println("✅ Data saved to $filepath")
end


function plotSSB_Data(; filepath = joinpath("Results/Data", "PhaseDiagram.csv"))
    df = CSV.read(filepath, DataFrame)

    # Convert data to grid format for heatmap
    μ0s = sort(unique(df.μ0))
    λs = sort(unique(df.λ))
    Z = [df.value[(df.μ0 .== μ0) .& (df.λ .== λ)][1] for λ in λs, μ0 in μ0s]

    plt = heatmap(
        μ0s, λs, Z,
        xlabel = "μ₀^2",
        ylabel = "λ",
        colorbar_title = "<ϕ²>",
        title = "Phase diagram: ⟨ϕ²⟩ Grid",
    )

    # Ensure Data directory exists
    if !isdir("Results/Plots")
        mkpath("Results/Plots")
    end


    plotpath = joinpath("Results/Plots", "PhaseDiagram.png")
    savefig(plt, plotpath)
    println("✅ Plot saved to $plotpath")
    return plt
end

getSSB_Data()
plotSSB_Data()
