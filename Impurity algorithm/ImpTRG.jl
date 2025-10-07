using TNRKit
using TensorKit
import TNRKit: run!
import TNRKit: stopcrit
import TNRKit: TNRScheme
import TNRKit: SVD12


mutable struct ImpTRG <: TNRScheme
    "central tensor"
    Tpure::TensorMap
    T1::TensorMap
    T2::TensorMap
    T3::TensorMap
    T4::TensorMap

    "finalization function"
    finalize!::Function
    function ImpTRG(Tpure::TensorMap{E, S, 2, 2}, T1::TensorMap{E, S, 2, 2},T2::TensorMap{E, S, 2, 2}, 
        T3::TensorMap{E, S, 2, 2}, T4::TensorMap{E, S, 2, 2}; finalize = (finalize!)) where {E, S}
        return new(Tpure, T1, T2, T3, T4, finalize)
    end
end


function step!(scheme::ImpTRG, trunc::TensorKit.TruncationScheme)
    """
    Gives 4 impurity tensors and 1 rest of lattice tensor in. Yields the 4 new tensors of one iterationstep of TRG.

    Input order tensors:
        1------>2
        ^       |
        |       |
        |       |
        4<------3

    Output order tensors:
            1
             
        4       2
             
            3
    """
    
    # Tensor1
    A1, B1 = SVD12(scheme.T1, trunc)
    
    # Tensor2
    tensor2p = transpose(scheme.T2, ((2, 4), (1, 3)))
    C2, D2 = SVD12(tensor2p, trunc)

    # Tensor3
    A3, B3 = SVD12(scheme.T3, trunc)
    
    # Tensor4
    tensor4p = transpose(scheme.T4, ((2, 4), (1, 3)))
    C4, D4 = SVD12(tensor4p, trunc)

    # Tensor pure
    Ap, Bp = SVD12(scheme.Tpure, trunc)
    tensorpurep = transpose(scheme.Tpure, ((2, 4), (1, 3)))
    Cp, Dp = SVD12(tensorpurep, trunc)

    # Debug
    #println(space(Dp, 1), "  ", space(Dp, 2))
    #println(space(B1, 1), "  ", space(B1, 2))
    #println(space(C2, 1), "  ", space(C2, 2))
    #println(space(Ap, 1), "  ", space(Ap, 2))


    # Contract
    @planar scheme.Tpure[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar scheme.T1[-1, -2; -3, -4] := Dp[-2; 1 2] * B1[-1; 4 1] * C2[4 3; -3] * Ap[3 2; -4]
    @planar scheme.T2[-1, -2; -3, -4] := D2[-2; 1 2] * B3[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar scheme.T3[-1, -2; -3, -4] := D4[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * A3[3 2; -4]
    @planar scheme.T4[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * C4[4 3; -3] * A1[3 2; -4]
end


function finalize!(scheme::ImpTRG)
    # First normalize everything by the pure tensor
    npure = norm(@tensor scheme.Tpure[1 2; 2 1])
    scheme.T1 /= npure
    scheme.T2 /= npure
    scheme.T3 /= npure
    scheme.T4 /= npure
    scheme.Tpure /= npure

    # Then calculate the contracted/traced 4 impurity tensors
    nimp = norm(@tensoropt scheme.T1[5 4;6 1] * scheme.T2[1 2;7 5] * scheme.T3[3 7;2 8] * scheme.T4[8 6;4 3])

    return npure, nimp
end



function run!(
        scheme::ImpTRG, trscheme::TensorKit.TruncationScheme, criterion::stopcrit;
        finalize_beginning = true
    )

    data_npure = []
    data_nimp = []


    println("Starting simulation\n $(scheme)\n")
    if finalize_beginning
        npure, nimp = scheme.finalize!(scheme)
        push!(data_npure, npure)
        push!(data_nimp, nimp)
    end

    steps = 0
    crit = true

    t = @elapsed while crit
        step!(scheme, trscheme)
        npure, nimp = scheme.finalize!(scheme)
        push!(data_npure, npure)
        push!(data_nimp, nimp)
        steps += 1
        crit = criterion(steps, data_npure)
    end

    println("Simulation finished\n $(stopping_info(criterion, steps, data_npure))\n Elapsed time: $(t)s\n Iterations: $steps")
    
    return data_npure, data_nimp
end



# Helper SVD functions

function SVD12(T::AbstractTensorMap{E, S, 2, 2}, trunc::TensorKit.TruncationScheme) where {E, S}
    U, s, V, e = tsvd(T; trunc = trunc)
    return U * sqrt(s), sqrt(s) * V
end


function stopping_info(::maxiter, steps::Int, data)
    return "Maximum amount of iterations reached: $(steps)"
end