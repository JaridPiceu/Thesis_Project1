using Plots
using CSV, DataFrames
using TNRKit
using TensorKit

include("../TensorMaker/TensorFactory.jl")

# Reproducing figure 1 of Kadoh (http://arxiv.org/abs/1811.12376)


# Parameters
K = 16 #In paper: 256
μ0s = [-0.1841626, -0.1006174, -0.06635522, -0.02395010, -0.01280510]
λs = [1.0, 0.5, 0.0312, 0.01, 0.005]

function getSVs(μ0, λ)
    T = getTensor(K, μ0, λ, 0)
    _, S, _ = tsvd(T)
    SVs = [S[i, i] / S[1, 1] for i in 1:K^2] #Normalized to the largest
    return SVs
end


# Initialize the plot
plt = scatter(xlims=(0, 25), xlabel="Singular value index i", ylabel="σ_i / σ_1", title="Normalized Singular Values (K=$K)", yscale=:log10)

ns = collect(range(1, K^2))
# Loop over parameters
for i in 1:5
    SVs = getSVs(μ0s[i], λs[i])
    # Add to plot
    scatter!(plt, ns, SVs, label="μ0=$(μ0s[i]), λ=$(λs[i])")
end

# Save the plot
filepath = joinpath("Results/Plots", "SV_fmatrix.png")
savefig(plt, filepath)