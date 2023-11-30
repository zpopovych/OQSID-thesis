using LinearAlgebra

function hankel(y::AbstractArray)
    m, time_duration = size(y) # m - dimention of output vector y, time_duration - length of timeseries (number of time steps)
    q = Int(round(time_duration/2)) # q - is the size of Hankel matrix 
    H = zeros(eltype(y), q * m , q) 
    for r = 1:q, c = 1:q # r - rows, c -columns
        H[(r-1)*m+1:r*m, c] = y[:, r+c-1]
    end
    return H, m
end

function ERA(Y::AbstractArray, r) 
    # y - output time series dim[y] = m x number_of_time_steps
    # r - rank of the system we want to identify
    
    H, m = hankel(Y) # Hankel matrix H, and dimention of output m
    U, Σ, V⁺ = svd(H) # Singular value decomposition of H to U,  Σ,  V†
    
    s = Diagonal(sqrt.(Σ)) # Matrix square root 
    U = U * s
    V⁺ = s * V⁺
      
    C = U[1:m, 1:r] # m - dimention of output, r - rank of the system
    
    U₊ = U[1:end-m, :] # U↑
    U₋ = U[m+1:end, :] # U↓
    
    A = pinv(U₊) * U₋
    A = A[1:r, 1:r] # r - estimated rank of the system
    
    x₀⁽¹⁾ = pinv(U) * H
    x₀⁽¹⁾ = x₀⁽¹⁾[1:r, 1]
    
    x₀⁽²⁾= V⁺[1:r, 1]
    
    Y₀ = Y[:,1]

    x₀⁽³⁾ = zeros(r) 

    try
        x₀⁽³⁾  = (C\Y₀)[1:r, 1]
    end

    x₀ᵒᵖᵗⁱᵒⁿˢ = [x₀⁽¹⁾, x₀⁽²⁾, x₀⁽³⁾]
    
    norms = [norm(Y₀ - C*x₀⁽ⁱ⁾) for x₀⁽ⁱ⁾ in x₀ᵒᵖᵗⁱᵒⁿˢ]
   
    x₀ = x₀ᵒᵖᵗⁱᵒⁿˢ[argmin(norms)]
    
    return A, C, x₀, Σ

end