using TensorKit
using FastGaussQuadrature


function f(ϕ1, ϕ2, μ0, λ, h=0)
    return exp(
        -1/2 * (ϕ1-ϕ2)^2
        - μ0/8 * (ϕ1^2+ϕ2^2)
        - λ/16 * (ϕ1^4+ϕ2^4)    
        + h/4 *  (ϕ1+ϕ2)
    )
end 

function fmatrix(ys, μ0, λ, h=0)
    K = length(ys)
    matrix = zeros(K, K)
    for i in 1:K
        for j in 1:K
            matrix[i, j] = f(ys[i], ys[j], μ0, λ, h)        
        end
    end
    return TensorMap(matrix, ℂ^K ← ℂ^K)
end


function getTensor(K, μ0, λ, h=0)
    # Weights and locations
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix(ys, μ0, λ, h)

    # SVD fmatrix
    U, S, V = tsvd(f)

    # Make tensor for one site
    T_arr = [
        sum(
            √(S[i,i] * S[j,j] * S[k,k] * S[l,l]) *
            ws[p] * exp(ys[p]^2) *
            U[p,i] * U[p,j] * V[p,k] * V[p,l]
            for p in 1:K
        )
        for i in 1:K, j in 1:K, k in 1:K, l in 1:K
    ]

    T = TensorMap(T_arr, ℂ^K ⊗ ℂ^K ← ℂ^K ⊗ ℂ^K)
    return T
end 