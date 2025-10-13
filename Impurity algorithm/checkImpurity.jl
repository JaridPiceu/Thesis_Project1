include("TensorFactory.jl")
include("testimpurity.jl")

using TNRKit
using TensorKit

function getTRG()
    T = getTensor(2,1,1,1) # partition function of classical Ising model at the critical point
    scheme = TRG(T) # Bond-weighted TRG (excellent choice)
    data = run!(scheme, truncdim(16), maxiter(1); finalize_beginning=false)
    return data, scheme.T
end

function getImpurityTRG()
    T = getTensor(2,1,1,1)
    T1, T2, T3, T4 = stepImpurity(T, T, T, T, T, truncdim(16))
    n_pure, n_imp, T1, T2, T3, T4 = normalizeAll(T1, T2, T3, T4, T)
    return n_pure, n_imp, T1
end

getTRG()
getImpurityTRG()

function check()
    normTRG, tensorTRG = getTRG()
    n_pure, n_imp, tensorImpurity = getImpurityTRG()

    @assert normTRG[1] ≈ n_imp "Norms do not match"
    @assert tensorTRG ≈ tensorImpurity "Tensors do not match"
    println("Succeed!")
end

check()