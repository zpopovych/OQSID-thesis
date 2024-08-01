module LiPoSID

using LinearAlgebra
using QuantumOptics

using DynamicPolynomials
# using MomentTools
#using MosekTools
using Random
using JuMP
using NLopt
#using TSSOS
#using Clustering
using HDF5

#using HomotopyContinuation

function hankel(y::AbstractArray)
    m, time_duration = size(y) # m - dimention of output vector y, time_duration - length of timeseries (number of time steps)
    q = Int(round(time_duration/2)) # q - is the size of Hankel matrix 
    H = zeros(eltype(y), q * m , q) 
    for r = 1:q, c = 1:q # r - rows, c -columns
        H[(r-1)*m+1:r*m, c] = y[:, r+c-1]
    end
    return H, m
end

function lsid_ACx0(Y::AbstractArray, Δt) #, δ = 1e-6)
    # y - output time series dim[y] = m x number_of_time_steps
    # δ - precission cutoff all the smaller values of Σ will be discarded 
    
    H, m = hankel(Y) # Hankel matrix and dimention of output (should be 12 in our case)
    U, Σ, Vd = svd(H) # Singular value decomposition of H to U,  Σ,  V†
    
    s = Diagonal(sqrt.(Σ)) # Matrix square root 
    U = U * s
    Vd = s * Vd
     
    # n = argmin(abs.(Σ/maximum(Σ) .- δ)) - 1 # estimated rank of the system

    Sigma_log = log.(Σ/maximum(Σ))
    Sigma2D = reshape(Sigma_log, (1, length(Sigma_log)))

    n = minimum(counts(kmeans(Sigma2D, 2))) + 1
    
    C = U[1:m, 1:n] # m -dimention of output, n - rank of the system
    
    U_up = U[1:end-m, :] # U↑
    U_down = U[m+1:end, :] # U↓
    
    A = pinv(U_up) * U_down
    # Ac = log(A)/Δt 
    # Ac = Ac[1:n, 1:n] 
    A = A[1:n, 1:n] # n - estimated rank of the system
    
    x0 = pinv(U) * H
    x0 = x0[1:n, 1]
    
    return A, C, x0 # was A, Ac, C, x0

end

function lsid_n_ACx0(Y::AbstractArray, Δt, n) 
    # y - output time series dim[y] = m x number_of_time_steps
    # n - rank of the system we want to identify
    
    H, m = hankel(Y) # Hankel matrix and dimention of output (should be 12 in our case)
    U, Σ, Vd = svd(H) # Singular value decomposition of H to U,  Σ,  V†
    
    s = Diagonal(sqrt.(Σ)) # Matrix square root 
    U = U * s
    Vd = s * Vd
      
    C = U[1:m, 1:n] # m -dimention of output, n - rank of the system
    
    U_up = U[1:end-m, :] # U↑
    U_down = U[m+1:end, :] # U↓
    
    A = pinv(U_up) * U_down
    # Ac = log(A)/Δt 
    # Ac = Ac[1:n, 1:n] 
    A = A[1:n, 1:n] # n - estimated rank of the system
    
    x0 = pinv(U) * H
    x0 = x0[1:n, 1]
    
    return A, C, x0 # was A, Ac, C, x0

end

function lsid_n_ACx0Σ(Y::AbstractArray, Δt, n) 
    # y - output time series dim[y] = m x number_of_time_steps
    # n - rank of the system we want to identify
    
    H, m = hankel(Y) # Hankel matrix and dimention of output (should be 12 in our case)
    U, Σ, Vd = svd(H) # Singular value decomposition of H to U,  Σ,  V†
    
    s = Diagonal(sqrt.(Σ)) # Matrix square root 
    U = U * s
    Vd = s * Vd
      
    C = U[1:m, 1:n] # m -dimention of output, n - rank of the system
    
    U_up = U[1:end-m, :] # U↑
    U_down = U[m+1:end, :] # U↓
    
    A = pinv(U_up) * U_down
    # Ac = log(A)/Δt 
    # Ac = Ac[1:n, 1:n] 
    A = A[1:n, 1:n] # n - estimated rank of the system
    
    x0_1 = pinv(U) * H
    x0_1 = x0_1[1:n, 1]
    
    x0_2= Vd[1:n, 1]
    
    # Y0 = [1.0   0.0  -0.0   0.0   0.0  -0.0   0.5   0.5  -0.0   0.5   0.0   0.5 ]'
    Y0 = Y[:,1]

    try
        global x0_3 = (C\Y0)[1:n, 1]
    catch
        global x0_3 = zeros(n)
    end

    x0_list = [x0_1, x0_2, x0_3]
       
    norms = [norm(Y0 - C*x0_i) for x0_i in x0_list]
   
    x0 = x0_list[argmin(norms)]
    
    return A, C, x0, Σ

end


function lsid_n_ACx0Σ_old(Y::AbstractArray, Δt, n) 
    # y - output time series dim[y] = m x number_of_time_steps
    # n - rank of the system we want to identify
    
    H, m = hankel(Y) # Hankel matrix and dimention of output (should be 12 in our case)
    U, Σ, Vd = svd(H) # Singular value decomposition of H to U,  Σ,  V†
    
    s = Diagonal(sqrt.(Σ)) # Matrix square root 
    U = U * s
    Vd = s * Vd
      
    C = U[1:m, 1:n] # m -dimention of output, n - rank of the system
    
    U_up = U[1:end-m, :] # U↑
    U_down = U[m+1:end, :] # U↓
    
    A = pinv(U_up) * U_down
    # Ac = log(A)/Δt 
    # Ac = Ac[1:n, 1:n] 
    A = A[1:n, 1:n] # n - estimated rank of the system
    
    x0 = pinv(U) * H
    x0 = x0[1:n, 1]
    
    return A, C, x0, Σ

end

function propagate(A, x0, steps)
    
    x = []
    push!(x, x0)

    @assert size(x0,1) == size(A,1) == size(A,2)

    for i=2:steps
        push!(x, A * x[end])
    end

    return x
end 

function propagate(A, C, x0, steps)
    n = size(A, 1)
    @assert size(x0,1) == n
    y = zeros(size(C,1), steps) 
    xₜ = x0
    for t in 1:steps
        y[:, t] = C * xₜ
        xₜ = A * xₜ
    end
    return y
end 

function propagate_LTI(A, C, x₀, n, steps)
    @assert n <= size(A,  1)
    @assert n <= size(x₀, 1)
    y = zeros(size(C,1), steps) 
    xₜ = x₀[1:n]
    for t in 1:steps
        y[:, t] = C[:,1:n] * xₜ
        xₜ = A[1:n,1:n] * xₜ
    end
    return y
end 

function rand_dm(n)
    # return a random density matrix
    ρ = -1 .+ 2 * rand(n, n) 
    ρ += im * (-1 .+ 2 * rand(n, n))  
    ρ = ρ * ρ'
    Hermitian(ρ / tr(ρ))
end

function rand_herm(n)
    # return a random hermitian matrix
    h = -1 .+ 2 * rand(n, n)
    h += im *(-1 .+ 2 *  rand(n, n))
    h = 0.5 * (h + h')
    Hermitian(h)
end

function bloch(ρ::Matrix{ComplexF64})
    # Pauli matricies
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]   
    #bloch_vec = [real(tr(σᵢ * ρ)) for σᵢ in σ[2:end]]
    bloch_vec = [real(tr(σᵢ * ρ)) for σᵢ in σ[1:3]]
end

function bloch(ρ_list::Union{Vector{Any},Vector{Matrix{ComplexF64}}})
    # ρ_list::Vector{Matrix{ComplexF64}}
    # Pauli matricies
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]
    
    time_steps = length(ρ_list)
    bloch_vec = zeros(3, time_steps)
    for t in 1:time_steps
        bloch_vec[:, t] = [real(tr(σ[i] * ρ_list[t])) for i=1:3] # 2 ???
    end
    convert.(Float64, bloch_vec)
end

function bloch(ρ_list::Vector{Hermitian{ComplexF64, Matrix{ComplexF64}}})
    # Pauli matricies
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]
    time_steps = length(ρ_list)
    bloch_vec = zeros(3, time_steps)
    for t in 1:time_steps
        bloch_vec[:, t] = [real(tr(σ[i] * ρ_list[t])) for i=1:3] # 2 ???
    end
    convert.(Float64, bloch_vec)
end

function rho_from_bloch(bloch_vec::Vector{Float64})
    # Pauli matricies
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]
    ρ = (sum([bloch_vec[i] * σ[i] for i=1:3]) + I)/2 
    ρ ::Matrix{ComplexF64}
end

function rho_series_from_bloch(bloch_vec::Matrix{Float64})
    time_steps = size(bloch_vec, 2)
    ρ = Vector{Matrix{ComplexF64}}() # size !!!
    for t in 1:time_steps
        push!(ρ, rho_from_bloch(bloch_vec[:, t]))     
    end
    ρ ::Vector{Matrix{ComplexF64}}
end

function rho3d_from_bloch(bloch_vec::Matrix{Float64})
    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]
    time_steps = size(bloch_vec, 2)
    ρ = zeros(2, 2, time_steps) + im*zeros(2, 2, time_steps)
    for t in 1:time_steps
        ρ[:, :, t] = (sum([bloch_vec[i, t] * σ[i] for i=1:3]) + I)/2       
    end
    ρ ::Array{ComplexF64, 3}
end

function rand_Linblad_w_noise(basis, seed, w, t_list)
    # seed - to generate reproducable system,
    # w - noise level
    # t_list - time span
    
    # basis = NLevelBasis(2) # define 2-level basis

    n = basis.N

    Random.seed!(seed)    
    
    ρ₀ = DenseOperator(basis, rand_dm(n))  # initial state density matrix
    H = DenseOperator(basis, rand_herm(n)) # Hamiltonian of the system
    J = DenseOperator(basis, (-1 .+ 2 *randn(n, n)) + im*(-1 .+ 2 *randn(n, n))) # Lindblad decipator  was rand !!!!!!
    
    time, ρ_exact = timeevolution.master(t_list, ρ₀, H, [J])

    ρ = [ (1 - w) * ρₜ.data + w * rand_dm(n) for ρₜ in ρ_exact ], H.data, J.data
       
end

function frobenius_norm2(m)
    return tr(m * m')
end

function lindblad_rhs(ρ, H, J::Matrix)
    """
    Right hand side of the Lindblad master equation
    """
    return -im * (H * ρ - ρ * H) + J * ρ * J' - (J' * J  * ρ + ρ * J' * J) / 2
    
end

function lindblad_rhs(ρ, H, J::Array)
    """
    Right hand side of the Lindblad master equation with multiple dicipators
    """
   
    Σ = sum([ ( Jⱼ * ρ * Jⱼ' - (Jⱼ' * Jⱼ  * ρ + ρ * Jⱼ' * Jⱼ)/2 ) for Jⱼ in J ])
    
    return -im * (H * ρ - ρ * H) + Σ 
    
end


function lindblad_rhs(ρ, H, J::Array, g)
    """
    Right hand side of the Lindblad master equation with multiple dicipators
    """
   
    Σ = sum([ ( Jⱼ * ρ * Jⱼ' - (Jⱼ' * Jⱼ  * ρ + ρ * Jⱼ' * Jⱼ)/2 ) for Jⱼ in J ])
    
    return -im * (H * ρ - ρ * H) + g * Σ 
    
end

import Base.real
function real(p::AbstractPolynomial)
    sum(real(coef) * mon for (coef, mon) in zip(coefficients(p), monomials(p))) #if ~isapproxzero(abs(coef)))
end

function pade_obj(ρ::Array{ComplexF64,3}, t, H, J)
   
    obj = 0
    for i in 2:size(ρ,3)
        obj += frobenius_norm2(
            ρ[:, :, i] - ρ[:, :, i-1] 
            - (t[i]-t[i-1])*lindblad_rhs((ρ[:, :, i]+ρ[:, :, i-1])/2, H, J)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function pade_obj(ρ::Vector{Matrix{ComplexF64}}, t::Vector{Float64}, H, J)

    obj = 0
    for i in 2:size(ρ,1)
        obj += frobenius_norm2(
            ρ[i] - ρ[i-1] 
            - (t[i]-t[i-1])*lindblad_rhs((ρ[i]+ρ[i-1])/2, H, J)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function simpson_obj(ρ::Vector{Matrix{ComplexF64}}, t, H, J)
    
    obj = 0
    for i in 3:length(ρ)
        obj += frobenius_norm2(
            ρ[i] - ρ[i-2] - (t[i]-t[i-1])lindblad_rhs((ρ[i-2] + 4ρ[i-1] + ρ[i])/3, H, J)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function simpson_obj(ρ::Vector{Matrix{ComplexF64}}, t, H, J, g)
    
    obj = 0
    for i in 3:length(ρ)
        obj += frobenius_norm2(
            ρ[i] - ρ[i-2] - (t[i]-t[i-1])lindblad_rhs((ρ[i-2] + 4ρ[i-1] + ρ[i])/3, H, J, g)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function simpson_obj(ρ::Array{ComplexF64,3}, t, H, J)  
    
    obj = 0
    for i in 3:length(ρ)
        obj += frobenius_norm2(
            ρ[:, :, i] - ρ[:, :, i-2] - (t[i]-t[i-1])lindblad_rhs((ρ[:, :, i-2] + 4ρ[:, :, i-1] + ρ[:, :, i])/3, H, J)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function simpson_obj_g(ρ::Array{ComplexF64,3}, t, H, J, g)  
    
    obj = 0
    for i in 3:length(ρ)
        obj += frobenius_norm2(
            ρ[:, :, i] - ρ[:, :, i-2] - (t[i]-t[i-1])lindblad_rhs((ρ[:, :, i-2] + 4ρ[:, :, i-1] + ρ[:, :, i])/3, H, J, g)
        )
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function boole_obj(ρ::Vector{Matrix{ComplexF64}}, t, H, A)
    
    obj = 0
    
    for i in 5:length(ρ)
        ρᵇᵒᵒˡ = 2(7ρ[i-4] + 32ρ[i-3] + 12ρ[i-3] + 32ρ[i-2] + 7ρ[i])/45  
        obj += frobenius_norm2( ρ[i] - ρ[i-4] - (t[i]-t[i-1])lindblad_rhs(ρᵇᵒᵒˡ, H, A) )
    end
    
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    
    return obj

end

function kraus_obj(ρ::Vector{Matrix{ComplexF64}}, K1, K2) 
    obj = 0
    for i in 1:length(ρ)-1
        obj += frobenius_norm2(K1 * ρ[i] * K1' - ρ[i+1]) + frobenius_norm2(K2 * ρ[i] * K2' - ρ[i+1])
    end
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    return obj
end

function kraus_obj(ρ, K) 
    obj = 0
    for i in 1:length(ρ)-1
        obj += LiPoSID.frobenius_norm2(sum(k * ρ[i] * k' for k in K) - ρ[i+1])
    end
    return real(obj)
end

function kraus_obj_constr(ρ, K) 
    obj = 0
    for i in 1:length(ρ)-1
        obj += frobenius_norm2(sum(k * ρ[i] * k' for k in K) - ρ[i+1])
    end
    constr = frobenius_norm2(sum(k' * k for k in K) - I)
    return real(obj), real(constr)*1e3
end

function timeevolution_kraus(t_steps, ρ₀, K)

    K = [convert.(ComplexF64, k) for k in K]

    ρ = [ρ₀]
    for t = 2:t_steps
        #push!(ρ, Hermitian(sum([K[i]* ρ[end] * K[i]' for i = 1:length(K)])))
        ρ_next = Hermitian(sum(K[i]* ρ[end] * K[i]' for i = 1:length(K)))
        push!(ρ, ρ_next/tr(ρ_next))
    end
    return ρ
end  

function rand_Kraus_w_noise(seed, w, time_span)
    Random.seed!(seed)
    
    ρ₀ = LiPoSID.rand_dm(2)     

    K1 = rand(2,2) + im*rand(2,2)
    K2 = rand(2,2) + im*rand(2,2)
    
    ρ_exact = timeevolution_kraus(time_span, ρ₀, [K1, K2])
    
    ρ = [ (1 - w) * ρₜ + w * LiPoSID.rand_dm(2) for ρₜ in ρ_exact ]
end

function rand_Kraus_w_noise(seed, w, time_span, kraus_rank)
    Random.seed!(seed)
    
    ρ₀ = LiPoSID.rand_dm(2)
    
    K = [rand(2,2) + im*rand(2,2) for i in 1:kras_rank]
    
    ρ_exact = timeevolution_kraus(time_span, ρ₀, K)
    
    ρ = [ (1 - w) * ρₜ + w * LiPoSID.rand_dm(2) for ρₜ in ρ_exact ] # adding white noise
end


# using NLopt

function minimize_local(obj, guess) # polynomial objective, and guess x candidate
    vars = variables(obj)
    
    @assert length(vars) == length(guess)

    function g(a...)
        # Converting polynomial expression to function to be minimize
        obj(vars => a)
    end
    
    model = Model(NLopt.Optimizer)

    set_optimizer_attribute(model, "algorithm", :LD_MMA)
    
    #set_silent(model)
    @variable(model, y[1:length(vars)]);
    
    for (var, init_val) in zip(y, guess)
        set_start_value(var, init_val)
    end
    
    #= if length(constr_list) > 0
        @constraint(model, constr_list)
    end =# 
    
    register(model, :g, length(y), g; autodiff = true)
    @NLobjective(model, Min, g(y...))
    JuMP.optimize!(model)
    solution = vars => map(value, y)
    
    return solution
end 

function minimize_global(obj, constr_list = [])
    optimizer = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)
    obj_min, M = minimize(obj, constr_list, [], variables(obj), maxdegree(obj) ÷ 2, optimizer)
    
    r = get_minimizers(M)
    obj_min_vals = [obj(r[:,i]) for i=1:size(r)[2]]
    best_candidate = r[:, argmin(obj_min_vals)]
    
    minimize_local(obj, constr_list, best_candidate) 
   
end 

# using QuantumOptics

function quantum_series(basis, ρ)
    [ DenseOperator(basis, Hermitian(ρ[i])) for i = 1:length(ρ) ]
end

function fidelity_series(basis, ρ₁, ρ₂)

    #@assert  length(ρ₁) == length(ρ₂)

    len_of_series = min(length(ρ₁), length(ρ₂))

    ρ₁ = quantum_series(basis, ρ₁)
    ρ₂ = quantum_series(basis, ρ₂)

    return [abs(fidelity(ρ₁[i], ρ₂[i])) for i in 1:len_of_series]

end

function min_fidelity_between_series(basis, ρ1, ρ2)

    len_of_series = length(ρ1)

    @assert  length(ρ2) == len_of_series

    ρ1q = quantum_series(basis, ρ1)
    ρ2q = quantum_series(basis, ρ2)
    
    minimum([abs(fidelity(ρ1q[i], ρ2q[i])) for i in 1:len_of_series])

end

#using TSSOS

function min2step(obj, constr)
    # obj - is objective function
    # constr - one constraint in the form of equation
    
    # extract valiables from the objective
    vars = variables(obj)

    iter = 0
    best_sol = ones(length(vars))
    
    # Perform global minimization with TSSOS package
    try
        opt,sol,data = tssos_first([obj, constr], variables(obj), maxdegree(obj)÷2, numeq=1, solution=true, QUIET = true); 
    
        # execute higher levels of the TSSOS hierarchy
        iter = 1
        best_sol = sol

        while ~isnothing(sol)
            iter += 1
            best_sol = sol
            try
                opt,sol,data = tssos_higher!(data, solution=true, QUIET = true);
            catch
                break
            end
            
            if iter > 5
                best_sol = ones(length(vars))
                break
            end
        end
    catch
        best_sol = ones(length(vars))
    end  
   
    function g(a...)
        # Converting polynomial expression of objective to function to be minimized
        obj(vars => a)
    end
    
    function e(a...)
        # Converting polynomial expression of constraint to function to be minimize
        constr(vars => a)
    end
       
    # Create NLopt model
    model = Model(NLopt.Optimizer)

    # Set algorithm 
    set_optimizer_attribute(model, "algorithm", :LD_SLSQP) 
    
    # Set variables
    @variable(model, y[1:length(vars)]);

    # Register constraint
    register(model, :e, length(y), e; autodiff = true)
    
    @NLconstraint(model, e(y...) == 0)

    # Register objective
    register(model, :g, length(y), g; autodiff = true)
    @NLobjective(model, Min, g(y...))
    
    # Set guess
    guess = best_sol
    for (var, init_val) in zip(y, guess)
        set_start_value(var, init_val)
    end

    # Call JuMP optimization function
    JuMP.optimize!(model)

    solution = vars => map(value, y)

    return(solution, iter)
end  

function min2step(obj)
    # obj - is objective function

    # extract valiables from the objective
    vars = variables(obj)
    
    # Perform global minimization with TSSOS package
    iter = 0
    best_sol = ones(length(vars))

    try
        opt,sol,data = tssos_first(obj, variables(obj), solution=true, QUIET = true);
        # execute higher levels of the TSSOS hierarchy
        iter = 1
        best_sol = sol

        while ~isnothing(sol)
            iter += 1
            best_sol = sol
            try
                opt,sol,data = tssos_higher!(data, solution=true, QUIET = true);
            catch
                break
            end
            
            if iter > 5
                best_sol = ones(length(vars))
                break
            end
        end
    catch
        best_sol = ones(length(vars))
    end  
    
   
    function g(a...)
        # Converting polynomial expression of objective to function to be minimized
        obj(vars => a)
    end
       
    # Create NLopt model
    model = Model(NLopt.Optimizer)

    # Set algorithm 
    set_optimizer_attribute(model, "algorithm", :LD_SLSQP) 
    
    # Set variables
    @variable(model, y[1:length(vars)]);

    # Register objective
    register(model, :g, length(y), g; autodiff = true)
    @NLobjective(model, Min, g(y...))
    
    # Set guess
    guess = best_sol
    for (var, init_val) in zip(y, guess)
        set_start_value(var, init_val)
    end

    # Call JuMP optimization function
    JuMP.optimize!(model)

    solution = vars => map(value, y)

    return(solution, iter)
end  


function scaling_poly(p::Polynomial)
    X = transpose(hcat([exponents(t) for t in terms(p)]...))

    # Get the scaling via linear regression
    scaling = X \ log.(abs.(coefficients(p)))

    exp.(abs.(scaling))
end


"""
Try TSSOS, scaled TSSOS, and Homotopy Continuation to get the global minima of the polynomial
"""
function poly_min(p::Polynomial)

    # p = convert(Polynomial{true, Float64}, p)
    ################################################################################################
    #
    #   Try HomotopyContinuation
    #
    ################################################################################################

    # Find the critical points

    #minimizer_homotopy = nothing
    
    try 

        result = HomotopyContinuation.solve(differentiate.(p, variables(p)))
        critical_points = real_solutions(result)

        # Get the exact values for the exact objective function for the found critical points
        val_p = p.(critical_points)

        if length(critical_points) > 0
            global minimizer_homotopy = critical_points[argmin(val_p)]
        else global minimizer_homotopy = nothing
        end

    catch
        println(" Homotopy failed")
        global minimizer_homotopy = nothing
    #finally
        #minimizer_homotopy = nothing
    end

    #optimum = minimum(val_p)

    ################################################################################################
    #
    #   Try just plain TSSOS
    #
    ################################################################################################
    #minimizer_tssos = nothing
    try 
        opt,sol,data = tssos_first(p, variables(p), QUIET=true, solution=true, newton=false);
        previous_sol = sol

        while ~isnothing(sol)
            previous_sol = sol
            opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
        end

        global minimizer_tssos = previous_sol
    
    catch
        println(" TSSOS failed")
        global minimizer_tssos = nothing
    #finally
        #minimizer_tssos = nothing
    end

    ################################################################################################
    #
    #   Try TSSOS on polynomial with scaled variables
    #
    ################################################################################################

    # find variable scaling
    scale = scaling_poly(p)

    # scale the polynomial
    p_scaled = subs(p, variables(p) => scale .* variables(p))

    # minimize
    # minimizer_scaled_tssos = nothing

    try
        opt,sol,data = tssos_first(p_scaled, variables(p), QUIET=true, solution=true, newton=false);
        previous_sol = sol

        while ~isnothing(sol)
            previous_sol = sol
            opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
        end

        global minimizer_scaled_tssos = scale .* previous_sol
    
    catch
        println("Scaled TSSOS failed")
        global minimizer_scaled_tssos = nothing
    #finally
        #minimizer_scaled_tssos = nothing
    end

    ################################################################################################
    #
    #   Comparing
    #
    ################################################################################################
    minimizers = [[minimizer_homotopy] [minimizer_tssos] [minimizer_scaled_tssos]]
    methods = ["homotopy" "tssos" "scaled_tssos"]
    sols_bits = .!isnothing.(minimizers)

    minimizers_found = minimizers[sols_bits]
    methods_found = methods[sols_bits]

    if length(minimizers_found) > 0
        val_p = p.(minimizers_found)
        best_indx = argmin(val_p)
        best_minimizer = minimizers_found[best_indx]
        best_method = methods_found[best_indx]

    else 
        print("All methods fail !!!")
        best_minimizer = ones(length(variables(p)))
        best_method = "fail"
    end

    # best_solution = minimize_local(p, best_minimizer)

    best_solution = variables(p) => best_minimizer

    return best_solution, best_method

end
                                                                                                                                             
function sos_min_newton(p::Polynomial)

    ################################################################################################
    #
    #   Try just plain TSSOS
    #
    ################################################################################################
    #minimizer_tssos = nothing
    try 
        opt,sol,data = tssos_first(p, variables(p), QUIET=true, solution=true);
        previous_sol = sol

        while ~isnothing(sol)
            previous_sol = sol
            opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
        end

        global minimizer_tssos = previous_sol
    
    catch
        println(" TSSOS failed")
        global minimizer_tssos = nothing
    #finally
        #minimizer_tssos = nothing
    end

    ################################################################################################
    #
    #   Try TSSOS on polynomial with scaled variables
    #
    ################################################################################################

    # devide by the largest coef

    pd = maximum(abs.(coefficients(p)))

    # find variable scaling
    scale = LiPoSID.scaling_poly(pd)

    # scale the polynomial
    p_scaled = subs(pd, variables(pd) => scale .* variables(pd))

    # minimize
    # minimizer_scaled_tssos = nothing

    try
        opt,sol,data = tssos_first(p_scaled, variables(p), QUIET=true, solution=true);
        previous_sol = sol

        while ~isnothing(sol)
            previous_sol = sol
            opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
        end

        global minimizer_scaled_tssos = scale .* previous_sol
    
    catch
        println("Scaled TSSOS failed")
        global minimizer_scaled_tssos = nothing
    #finally
        #minimizer_scaled_tssos = nothing
    end

    ################################################################################################
    #
    #   Comparing
    #
    ################################################################################################
    minimizers = [[minimizer_tssos] [minimizer_scaled_tssos]]
    methods = ["tssos" "scaled_tssos"]
    sols_bits = .!isnothing.(minimizers)

    minimizers_found = minimizers[sols_bits]
    methods_found = methods[sols_bits]

    if length(minimizers_found) > 0
        val_p = p.(minimizers_found)
        @show val_p                                                                                               
        best_indx = argmin(val_p)
        best_minimizer = minimizers_found[best_indx]
        best_method = methods_found[best_indx]

    else 
        print("All methods fail !!!")
        best_minimizer = ones(length(variables(p)))
        best_method = "fail"
    end

    # best_solution = minimize_local(p, best_minimizer)

    best_solution = variables(p) => best_minimizer

    return best_solution, best_method

end                                                                                        
                                                                                    
                                                                                    
                                                                                    
function sos_min(p::Polynomial)

    ################################################################################################
    #
    #   Try just plain TSSOS
    #
    ################################################################################################
    #minimizer_tssos = nothing
    try 
        opt,sol,data = tssos_first(p, variables(p), QUIET=true, solution=true, newton=false);
        previous_sol = sol

        while ~isnothing(sol)
            previous_sol = sol
            opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
        end

        global minimizer_tssos = previous_sol
    
    catch
        println(" TSSOS failed")
        global minimizer_tssos = nothing
    #finally
        #minimizer_tssos = nothing
    end

    ################################################################################################
    #
    #   Try TSSOS on polynomial with scaled variables
    #
    ################################################################################################

    # find variable scaling
    scale = LiPoSID.scaling_poly(p)

    # scale the polynomial
    p_scaled = subs(p, variables(p) => scale .* variables(p))

    # minimize
    # minimizer_scaled_tssos = nothing

    try
        opt,sol,data = tssos_first(p_scaled, variables(p), QUIET=true, solution=true, newton=false);
        previous_sol = sol

        while ~isnothing(sol)
            previous_sol = sol
            opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
        end

        global minimizer_scaled_tssos = scale .* previous_sol
    
    catch
        println("Scaled TSSOS failed")
        global minimizer_scaled_tssos = nothing
    #finally
        #minimizer_scaled_tssos = nothing
    end

    ################################################################################################
    #
    #   Comparing
    #
    ################################################################################################
    minimizers = [[minimizer_tssos] [minimizer_scaled_tssos]]
    methods = ["tssos" "scaled_tssos"]
    sols_bits = .!isnothing.(minimizers)

    minimizers_found = minimizers[sols_bits]
    methods_found = methods[sols_bits]

    if length(minimizers_found) > 0
        val_p = p.(minimizers_found)
        best_indx = argmin(val_p)
        best_minimizer = minimizers_found[best_indx]
        best_method = methods_found[best_indx]

    else 
        print("All methods fail !!!")
        best_minimizer = ones(length(variables(p)))
        best_method = "fail"
    end

    # best_solution = minimize_local(p, best_minimizer)

    best_solution = variables(p) => best_minimizer

    return best_solution, best_method

end                                                                                  

function tssos_scaled(p::Polynomial)

    ################################################################################################
    #
    #   TSSOS on polynomial with scaled variables
    #
    ################################################################################################

    # find variable scaling
    scale = scaling_poly(p)

    # scale the polynomial
    p_scaled = subs(p, variables(p) => scale .* variables(p))

    # minimize
    # minimizer_scaled_tssos = nothing

    try
        opt,sol,data = tssos_first(p_scaled, variables(p), QUIET=true, solution=true, newton=false);
        previous_sol = sol

        while ~isnothing(sol)
            previous_sol = sol
            opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
        end

        global minimizer_scaled_tssos = scale .* previous_sol
    
    catch
        println("Scaled TSSOS failed")
        global minimizer_scaled_tssos = nothing
    end

    solution = variables(p) => minimizer_scaled_tssos

    return solution
end

function tssos_scaled(p::Polynomial, constr)

    ################################################################################################
    #
    #   TSSOS on polynomial with scaled variables
    #
    ################################################################################################

    # find variable scaling
    scale = scaling_poly(p)

    # scale the polynomial
    p_scaled = subs(p, variables(p) => scale .* variables(p))

    # minimize
    # minimizer_scaled_tssos = nothing

    try
        opt,sol,data = tssos_first([p_scaled, constr], variables(p), maxdegree(p)÷2, numeq=1, solution=true, QUIET = true); 
        previous_sol = sol

        while ~isnothing(sol)
            previous_sol = sol
            opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
        end

        global minimizer_scaled_tssos = scale .* previous_sol
    
    catch
        println("Scaled TSSOS failed")
        global minimizer_scaled_tssos = nothing
    end

    solution = variables(p) => minimizer_scaled_tssos

    return solution
end

#### HDF5 READING RESULTS ####

function get_seeds_and_timespan(file_name)   
    h5open(file_name,"r") do fid   # read file, preserve existing contents
        seeds = read(fid["seeds"])
        Δt = read(fid["dt"])
        tₘₐₓ = read(fid["t_max"])
        return seeds,  Δt, tₘₐₓ
    end
end

function get_noise_levels(file_name)   
    h5open(file_name,"r") do fid   # read file, preserve existing contents
        noise_levels = keys(fid["data_by_noise_level"])
        return noise_levels
    end
end

function get_variable_names(file_name, noise_level, seed)
        h5open(file_name,"r") do fid   # read file, preserve existing contents
        variable_names = keys(fid["data_by_noise_level"][string(noise_level)][string(seed)])
        return variable_names
    end
end

function get_by_name(file_name, var_name, noise_levels, seeds)
        h5open(file_name,"r") do fid # read file, preserve existing contents
        var_by_name = []
        for w in noise_levels
            current_noise_var = [ read(fid["data_by_noise_level"][string(w)][string(seed)][var_name]) for seed in seeds ]
            push!(var_by_name, current_noise_var)
        end
        return(var_by_name)
    end
end

function get_lsid(file_name, noise, seeds)
    A = get_by_name(file_name, "A", [noise], seeds)[1]
    C = get_by_name(file_name, "C", [noise], seeds)[1]
    x0 = get_by_name(file_name, "x0", [noise], seeds)[1]
    return A, C, x0
end

function get_kraus_sid(file_name, noise, seeds)  
    K1_sid = get_by_name(file_name, "K1_sid", [noise], seeds)[1]
    K2_sid = get_by_name(file_name, "K2_sid", [noise], seeds)[1]
    return K1_sid, K2_sid
end 

function get_lindblad_params(file_name, noise, key,  seeds, basis)
    H = [DenseOperator(basis, Hl) for Hl in get_by_name(file_name, "H_"*key, [noise], seeds)[1]]
    J = [DenseOperator(basis, Jl) for Jl in get_by_name(file_name, "J_"*key, [noise], seeds)[1]]
   return H, J
end

function lindblad_evolution(key, time_limit, Δt, noise_level, seed)
    time_span = [0:Δt:time_limit;]
    H_exact = DenseOperator(basis, get_by_name(file_name, "H_"*key, [noise_level], seed)[1][1])
    J_exact = DenseOperator(basis, get_by_name(file_name, "J_"*key, [noise_level], seed)[1][1])
    ρ0 = DenseOperator(basis, get_by_name(file_name, "rho0", [noise_level], seed)[1][1])
    time, ρ_exact_ser  = timeevolution.master(time_span, ρ0, H_exact, [J_exact])
    ρ = [ρₜ.data for ρₜ in ρ_exact_ser]
end

function lindblad_evolution_data(time_span, ρ0, H, J)
    time, ρ_ser  = timeevolution.master(time_span, ρ0, H, [J])
    ρ = [ρₜ.data for ρₜ in ρ_ser]
end


function Lindblad_time_evolution(basis, ρ0, time_span, H, A)
    
    H = convert.(ComplexF64, H)
    
    ρ0q = DenseOperator(basis, Hermitian(ρ0)) 

    Hq  = DenseOperator(basis, convert.(ComplexF64,H)) # reconstructed Hamiltonian of the system
    Aq = [ DenseOperator(basis, convert.(ComplexF64, A_i))  for A_i in A ]# reconstracted Lindblad decipator
    
    time, ρ_ser  = timeevolution.master(time_span, ρ0q, Hq, Aq)
    
    ρ = [ρₜ.data for ρₜ in ρ_ser]
    
    return ρ

end

function read_fidelity_table(file_name, fid_name, noise, seeds)
    fidelity_table = []
    h5open(file_name,"r") do fid # read file, preserve existing contents
        for seed in seeds
            push!(fidelity_table, read(fid[string(noise)][string(seed)][string(fid_name)]))
        end
        return(mapreduce(permutedims, vcat, fidelity_table))
    end
end

function get_rho_series(file_name, γ)
    h5open(file_name, "r") do file
        ρᵧ = read(file[string(γ)])
        t = ρᵧ["t"]
        ρ₀₀ = ρᵧ["p0"]; Re_ρ₀₁ = ρᵧ["s_re"];  Im_ρ₀₁ = ρᵧ["s_im"]
        ρ_series = []
        t_series = []

        for i in 1:length(t)
            ρᵢ= [ ρ₀₀[i]                      Re_ρ₀₁[i] + im * Im_ρ₀₁[i]
                  Re_ρ₀₁[i] - im * Im_ρ₀₁[i]  1 - ρ₀₀[i]                 ]
            push!(ρ_series, convert(Matrix{ComplexF64}, ρᵢ))
            push!(t_series, convert(Float64, t[i]))
        end
        return(ρ_series, t_series)
    end
end

function get_rho_series2(file_name, γ)
    h5open(file_name, "r") do file
        ρᵧ = read(file[string(γ)])
        t = ρᵧ["t"]
        ρ₀₀ = ρᵧ["p0"]; Re_ρ₀₁ = ρᵧ["s_re"];  Im_ρ₀₁ = ρᵧ["s_im"]
        ρ_series = []
        t_series = []

        for i in 1:length(t)
            ρᵢ= [ 1-ρ₀₀[i]                      Re_ρ₀₁[i] + im * Im_ρ₀₁[i]
                  Re_ρ₀₁[i] - im * Im_ρ₀₁[i]    ρ₀₀[i]                 ]
            push!(ρ_series, convert(Matrix{ComplexF64}, ρᵢ))
            push!(t_series, convert(Float64, t[i]))
        end
        return(ρ_series, t_series)
    end
end

function get_bosonic_bath_Lindblad_ME_model(file_name, γ, training_length )
    h5open(file_name,"r") do fid   # read file, preserve existing contents
        H = read(fid["gamma_"*string(γ)]["gt_"*string(training_length)]["H_sid_simp"])
        J_s3d = read(fid["gamma_"*string(γ)]["gt_"*string(training_length)]["J_sid_simp"])
        J_simp = [convert(Matrix{ComplexF64}, J_s3d[:, :, k]) for k in axes(J_s3d, 3)]
        return H,J_simp
    end
end   

function get_bosonic_bath_Kraus_map_model(file_name, γ, training_length )
    h5open(file_name,"r") do fid   # read file, preserve existing contents
        K3d = read(fid["gamma_"*string(γ)]["gt_"*string(training_length)]["K_sid"])
        K = [convert(Matrix{ComplexF64}, K3d[:, :, k]) for k in axes(K3d, 3)]
        return K
    end
end   

function get_bosonic_bath_data_coupling_levels(file_name)
    h5open(file_name,"r") do fid   # read file, preserve existing contents
        coupling_levels = keys(fid)
        return coupling_levels
    end
end

function get_bosonic_bath_models_coupling_levels(file_name)
    h5open(file_name,"r") do fid   # read file, preserve existing contents
        coupling_levels = [s[7:end] for s in keys(fid)]
        return coupling_levels
    end
end

function get_bosonic_bath_models_training_durations(file_name, coupling_level)
    h5open(file_name,"r") do fid   # read file, preserve existing contents
        coupling_levels = [s[4:end] for s in keys(fid["gamma_"*string(coupling_level)])]
        return coupling_levels
    end
end

function get_bosonic_bath_lsid(file, rank_group, γ_group)

    h5open(file,"r") do fid # read-only
        A = read(fid[rank_group][γ_group]["A"])
        # Ac = read(fid[γ_group]["Ac"])
        C = read(fid[rank_group][γ_group]["C"])
        x₀ = read(fid[rank_group][γ_group]["x0"])
        n = read(fid[rank_group][γ_group]["n"])
        Σ = read(fid[rank_group][γ_group]["sigma"])
        
        return A, C, x₀, n, Σ
    end
end

function get_operator(file, group, sub_group, operator_name)

    h5open(file,"r") do fid # read-only
        A = read(fid[group][sub_group][operator_name])
        return A
    end
end

function get_keys(df)
    h5open(df, "r") do file
        return keys(file)
    end
end

function get_cuts(df, γ)
    h5open(df, "r") do file
        return keys(file["gamma_"*string(γ)])
    end
end

function ⊗(A, B)
    return kron(A,B)
end
     
function obj_Lindblad_from_Kraus(K, Δt, H, J)

    n = size(K[1])[1]
    @assert size(H)[1] == size(J[1])[1] == n

    K = [convert.(ComplexF64, k) for k in K]
    
    A = sum( transpose(k') ⊗ k for k in K )  
    Lᴷʳᵃᵘˢᵥ = log(A)/Δt
    
    𝓘 = I(n) * 1.   
    
    U = -im*(𝓘 ⊗ H - transpose(H) ⊗ 𝓘)
    
    D = sum( 2*transpose(j')⊗j - 𝓘⊗(j'*j) - transpose(j)*transpose(j')⊗𝓘 for j in J )/2 
    
    Lᴸᴹᴱᵥ  = U + D
          
    ΔL = Lᴷʳᵃᵘˢᵥ - Lᴸᴹᴱᵥ
    
    obj = LiPoSID.frobenius_norm2(ΔL)
            
    obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
   
    return obj
end

function direct_DMD_01XY_b4_A(ρ)

    ρᵉ, ρᵍ, ρˣ, ρʸ = ρ
    lᵉ = length(ρᵉ); lᵍ = length(ρᵍ); lˣ = length(ρˣ); lʸ = length(ρʸ)
    lᵐᵃˣ = min(lᵉ, lᵍ,  lˣ, lʸ)  #choose time limit by shortest series
    bᵉ = LiPoSID.bloch(ρᵉ[1:lᵐᵃˣ])
    bᵍ = LiPoSID.bloch(ρᵍ[1:lᵐᵃˣ])
    bˣ = LiPoSID.bloch(ρˣ[1:lᵐᵃˣ])
    bʸ = LiPoSID.bloch(ρʸ[1:lᵐᵃˣ])
    Yᵉ = [bᵉ; ones(lᵉ)'] # augmented Bloch 4D vectors
    Yᵍ = [bᵍ; ones(lᵍ)']
    Yˣ = [bˣ; ones(lˣ)']
    Yʸ = [bʸ; ones(lʸ)']

    Yᵉ⁻ = Yᵉ[:,1:end-1]; Yᵉ⁺ = Yᵉ[:,2:end]
    Yᵍ⁻ = Yᵍ[:,1:end-1]; Yᵍ⁺ = Yᵍ[:,2:end]
    Yˣ⁻ = Yˣ[:,1:end-1]; Yˣ⁺ = Yˣ[:,2:end]
    Yʸ⁻ = Yʸ[:,1:end-1]; Yʸ⁺ = Yʸ[:,2:end]

    Y⁻ = hcat(Yᵉ⁻, Yᵍ⁻, Yˣ⁻, Yʸ⁻)
    Y⁺ = hcat(Yᵉ⁺, Yᵍ⁺, Yˣ⁺, Yʸ⁺)

    A = Y⁺ * pinv(Y⁻)

    return A

end

function Kraus(ρ₀, E)
    ρ = sum(K * ρ₀ * K' for K in E)
    ρ = ρ/tr(ρ)
    return Hermitian(ρ)
end

function choi(ρᵍ, ρᵉ, ρˣ, ρʸ)
    ρ₁ = ρᵍ
    ρ₄ = ρᵉ
    ρ₂ = ρˣ + im*ρʸ - (1+im)*(ρ₁+ρ₄)/2; # this matrix is not Hermitian
    ρ₃ = ρˣ - im*ρʸ - (1-im)*(ρ₁+ρ₄)/2; # this matrix is not Hermitian

    σₓ = [ 0  1
           1  0 ]  # X gate

    Λ = [ I   σₓ
        σₓ  -I ] / 2 # was -I in Niesen-Chuang (8.178)

    χ = Λ * [ ρ₁ ρ₂ 
            ρ₃ ρ₄ ] * Λ

    return χ
end


function operator_sum(χ)

    #@assert ishermitian(χ)

    σ = [ [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1] ]
    σˣ = σ[1]; σʸ = σ[2]; σᶻ = σ[3]; σᴵ = σ[4]; 

    E₀ = I(2)

    E₁ = σˣ  #  σₓ  or X gate
    E₂ = -im * σʸ
    E₃ = σᶻ

    Ẽ = [E₀, E₁, E₂, E₃] 

    d, U = eigen(χ)

    @assert U * diagm(d) * U' ≈ χ
    
    d = real.(d)

    E = []
    for i in 1:size(U)[2]
        if d[i] > 0 && d[i] ≉ 0
            Eᵢ = sqrt(d[i]) * sum(U[j,i] * Ẽ[j] for j in 1:size(U)[1])
            push!(E, Eᵢ)
        end
    end
    return E, d 
end

function QPT(ρᵍ, ρᵉ, ρˣ, ρʸ)

    χ = choi(ρᵍ, ρᵉ, ρˣ, ρʸ)

    E, d = operator_sum(χ)

    return E, d
end

function bloch4(ρ)

    b = convert.(Float64, [ ρ[1,2] + ρ[2,1],
                           (ρ[1,2] - ρ[2,1])*im,    #ρ[2,1] - ρ[1,2] ?
                            ρ[1,1] - ρ[2,2],
                               1                 ]) #ρ[1,1] + ρ[2,2]  

end

function dm_b4(b) 

    ρ = [ 1+b[3]         b[1]-im*b[2]
          b[1]+im*b[2]   1-b[3]       ]/2

end

function propagate_DMD_b4(A, ρ₀, lᵐᵃˣ)

    ρ = [ρ₀]

    for i in 2:lᵐᵃˣ
        push!(ρ, dm_b4( A * bloch4(ρ[end])))
    end

    return ρ

end

function DMD_step(A, ρ₀)
    dm_b4(A * bloch4(ρ₀))
end

function get_lindblad_operators(C::Matrix{ComplexF64}, basis_ops::Vector{Matrix{ComplexF64}})
    # Check that C is a square matrix and basis_ops has the same dimension
    n = size(C, 1)
    if size(C, 2) != n || length(basis_ops) != n
        throw(ArgumentError("Dimensions of C and basis_ops do not match"))
    end

    # Perform eigenvalue decomposition of C
    eigvals, eigvecs = eigen(C)

    # Construct the Lindblad operators
    lindblad_ops = []
    for i in 1:n
        if eigvals[i] > 1e-10  # Filter out negligible eigenvalues to ensure numerical stability
            lindblad_op = zeros(ComplexF64, size(basis_ops[1]))
            for j in 1:n
                lindblad_op .+= sqrt(eigvals[i]) * eigvecs[j, i] * basis_ops[j]
            end
            push!(lindblad_ops, lindblad_op)
        end
    end

    return lindblad_ops
end


end
