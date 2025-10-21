using Plots
using TensorKit
using CSV, DataFrames, Serialization

include("../TensorMaker/TensorFactory.jl")


"""
    getPhi4Data(K, μ0, λ; h=0, niter=15, ndimtrunc=16)

Runs the LoopTNR algorithm for the φ⁴ model tensor network and extracts CFT data.

# Arguments
- `K`: Coupling constant for the φ⁴ model.
- `μ0`: Mass parameter.
- `λ`: Coupling constant interaction strength.
- `h`: External field (default: 0).
- `niter`: Number of LoopTNR iterations (default: 15).
- `ndimtrunc`: Truncation dimension for tensors (default: 16).

# Returns
A tuple `(lnz, cft_data, central_charge)` where:
- `lnz`: Logarithm of the partition function (free energy).
- `cft_data`: CFT data.
- `central_charge`: Estimated central charge from CFT data.

# Notes
- Prints the elapsed time for the computation.
- Uses global variable `result` to store the output.
"""
function getPhi4Data(K, μ0, λ; h = 0, niter = 15, ndimtrunc = 16)
    t = @elapsed begin
        T = getTensor(K, μ0, λ, h)
        scheme = LoopTNR(T)
        data = run!(scheme, truncdim(ndimtrunc), maxiter(niter))

        lnz = free_energy(data, -1)
        cftdata = cft_data!(scheme, [sqrt(2), 2 * sqrt(2), 0])

        global result = (lnz, cftdata[Trivial()], cftdata["c"])
    end

    println("getPhi4Data completed in $(t) seconds")
    return result
end


"""
    getPhi4Data(K, μ0, λ; h=0, niter=15, ndimtrunc=16)

Runs the LoopTNR algorithm for the φ⁴ model tensor network and extracts CFT data. It stores the data in a file. If the request has been
    already calculated, it will look it up instead of calculating it again.

# Arguments
- `K`: Coupling constant for the φ⁴ model.
- `μ0`: Mass parameter.
- `λ`: Coupling constant interaction strength.
- `h`: External field (default: 0).
- `niter`: Number of LoopTNR iterations (default: 15).
- `ndimtrunc`: Truncation dimension for tensors (default: 16).

# Returns
A tuple `(lnz, trivial_data, central_charge, t)` where:
- `lnz`: Logarithm of the partition function (free energy).
- `trivial_data`: CFT data for the trivial sector.
- `central_charge`: Estimated central charge from CFT data.
- `t`: Computation time

# Notes
- Prints the elapsed time for the computation.
"""
function getPhi4Data_cached(K, μ0, λ; h = 0, niter = 15, ndimtrunc = 16)
    # Ensure the data directory exists
    pathdir = "Results/Data"
    pathfile = joinpath(pathdir, "phi4_cft.csv")

    # Initialize or load main registry
    if isfile(pathfile)
        df = CSV.read(pathfile, DataFrame)
    else
        df = DataFrame(
            K = Int[], μ0 = Float64[], λ = Float64[], h = Float64[],
            niter = Int[], ndimtrunc = Int[],
            lnz = Float64[], c = Float64[], dimsfile = String[], time = Float64[]
        )
    end

    # Check if we already computed this parameter combination
    existing = filter(
        row -> row.K == K &&
            row.μ0 == μ0 &&
            row.λ == λ &&
            row.h == h &&
            row.niter == niter &&
            row.ndimtrunc == ndimtrunc, df
    )

    if nrow(existing) > 0
        println("✅ Using cached result for (K=$K, μ0=$μ0, λ=$λ, h=$h, niter=$niter, ndimtrunc=$ndimtrunc)")
        dims_path = existing.dimsfile[1]
        dims = deserialize(dims_path)
        return existing.lnz[1], dims, existing.c[1], existing.time[1]
    end

    # Otherwise perform new computation
    println("⏳ Running new simulation for (K=$K, μ0=$μ0, λ=$λ, h=$h, niter=$niter, ndimtrunc=$ndimtrunc)...")
    t = @elapsed begin
        T = getTensor(K, μ0, λ, h)
        scheme = LoopTNR(T)
        data = run!(scheme, truncdim(ndimtrunc), maxiter(niter))
        lnz = free_energy(data, -1)
        cftdata = cft_data!(scheme, [sqrt(2), 2 * sqrt(2), 0])
        result = (lnz, cftdata[Trivial()], cftdata["c"])
    end

    println("✅ Simulation completed in $(round(t, digits = 2)) seconds")

    # Save dims separately as binary
    dims_dir = joinpath(pathdir, "CFTResults")
    dims_filename = "dims_K$(K)_μ$(μ0)_λ$(λ)_h$(h)_n$(niter)_tr$(ndimtrunc).jls"
    dims_path = joinpath(dims_dir, dims_filename)
    serialize(dims_path, result[2])

    # Register in the CSV
    push!(df, (K, μ0, λ, h, niter, ndimtrunc, result[1], result[3], dims_path, t))
    CSV.write(pathfile, df)

    return result[1], result[2], result[3], t
end


function plotCFT(K, μ0, λ; h = 0, niter = 15, ndimtrunc = 16)
    lnz, cft_data, c, t = getPhi4Data_cached(K, μ0, λ; h = h, niter = niter, ndimtrunc = ndimtrunc)
    
    # Plot the results
    xs = collect(range(1,length(cft_data)))
    plt = scatter(xs, real(cft_data)[2:end])   # Do not the first one (is zero)

    # Plot the exact results
    hline!(ising_cft_exact)
   
    # Save the plot
    filepath = joinpath("Results/Plots", "CFT_data.png")
    savefig(plt, filepath)
end

getPhi4Data_cached(16, -0.1, 0.1; h = 0, niter = 5, ndimtrunc = 16)

plotCFT(16, -0.1, 0.1; h = 0, niter = 5, ndimtrunc = 16)