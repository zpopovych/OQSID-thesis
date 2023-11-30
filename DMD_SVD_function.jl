using LinearAlgebra

function DMD_SVD(Y, r, Δt)   
    X₋ = Y[:,1:end-1]
    X₊ = Y[:,2:end]
    U, Σ, V = svd(X₋)
    Uʳ = U[:, 1:r]
    Σʳ = diagm(Σ[1:r])
    Vʳ = V[:, 1:r]
    Ã = Uʳ' * X₊ * Vʳ / Σʳ
    Λ, W = eigen(Ã)
    Φ = X₊ * Vʳ / Σʳ * W
    Ω = log.(Λ)/Δt
    x₁ = X₋[:,1]
    b₁ = Φ \ x₁
    return Φ, Ω, b₁, Λ
end  
