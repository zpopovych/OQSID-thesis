using LinearAlgebra

function hankel(y)
    m, time_duration = size(y) 
    q = Int(round(time_duration/2))  
    H = zeros(eltype(y), q * m , q) 
    for r = 1:q, c = 1:q 
        H[(r-1)*m+1:r*m, c] = y[:, r+c-1]
    end
    return H, m
end

function ERA(Y, r) 
    H, m = hankel(Y) # m - dimention of the output vector
    U, Σ, V⁺ = svd(H) 
    s = Diagonal(sqrt.(Σ)) 
    U = U * s
    V⁺ = s * V⁺
    C = U[1:m, 1:r] 
    U₊ = U[1:end-m, :] 
    U₋ = U[m+1:end, :] 
    A = pinv(U₊) * U₋
    A = A[1:r, 1:r] # r - estimated rank of the system
    x₀ = pinv(U) * H[:,1]
    x₀ = x₀[1:r, 1]   
    return A, C, x₀, Σ
end