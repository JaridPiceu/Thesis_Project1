include("ImpTRG.jl")
include("c:\\Users\\Jarid\\Documents\\4 Code\\Unief\\Thesis\\Project1\\TensorFactory.jl")
using Plots


function testDoesItRun()
    T = getTensor(2,1,1,1)
    scheme = ImpTRG(T, T, T, T, T)
    data_npure, data_nimp = run!(scheme, truncdim(16), maxiter(20); finalize_beginning=true)

    println(data_npure)
    println(data_nimp)
end
testDoesItRun()

function testIsingSameTRGAndImpTRG()
    β = 1
    T = getT(β)

    scheme_imp = ImpTRG(T, T, T, T, T)
    data_npure, data_nimp = run!(scheme_imp, truncdim(16), maxiter(15); finalize_beginning=true)
    fimp = free_energy(data_npure, β)

    scheme_trg = TRG(T)
    data_trg = run!(scheme_trg, truncdim(16), maxiter(15); finalize_beginning=true)
    ftrg = free_energy(data_trg, β)

    println("\n\nFree energy TRG: $ftrg")
    println("Free energy TRG: $fimp")
    @assert ftrg ≈ fimp "Free energy does NOT match"
end
testIsingSameTRGAndImpTRG()


function testIsingExpValue(niter=15, npoints=10)
    """
    Plot the expectation value <m^2> of the Ising model to check the results
    """
    βs = collect(range(0,3,npoints))
    Zs = []
    Z2s = []

    # Make first the impurity tensor
    σz = [1.0 0; 
            0 -1.0]  # ℂ^2 × ℂ^2

    
    for β in βs
        # Get the values for the normal partition function
        T = getT(β)
        scheme = TRG(T)
        data = run!(scheme, truncdim(16), maxiter(niter); finalize_beginning=true)
        Z = exp(free_energy(data, -1))
        push!(Zs, Z) 


        # Get the values for the Z_2 partition function
        #Suppose you want the expectation at the first index i (the top-left corner in your 4-site plaquette):
        T_exp = similar(T)
        for i in 1:2, j in 1:2, k in 1:2, l in 1:2
            T_exp[i,j,k,l] = σz[i,i] * T[i,j,k,l]
        end
        scheme_imp = ImpTRG(T, T_exp, T, T, T)
        data_npure, data_nimp = run!(scheme_imp, truncdim(16), maxiter(niter); finalize_beginning=true)
        Z2 = exp(free_energy(data_npure, -1))
        push!(Z2s, Z2) 
    end 

    ms = [Z2s[i]/Zs[i] for i in 1:length(Zs)]


    # Plot
    plot(βs, ms, xlabel="β", ylabel="⟨m⟩", title="Magnetization of 2D Ising model", legend=false)
    return βs, ms
end
testIsingExpValue(10)




function getT(β, h=0)
    
    function σ(i::Int64)
            return 2i - 3
    end
    T_array = Float64[
        exp(
                β * (σ(i)σ(j) + σ(j)σ(l) + σ(l)σ(k) + σ(k)σ(i)) +
                h / 2 * β * (σ(i) + σ(j) + σ(k) + σ(l))
            )
            for i in 1:2, j in 1:2, k in 1:2, l in 1:2
    ]

    T = TensorMap(T_array, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)
    return T
end
