module CtrlSID

include("../LiPoSID.jl")

using LinearAlgebra
#using QuantumOptics

using DynamicPolynomials
# using MomentTools
#using MosekTools
#using Random
using JuMP
using Ipopt
#using NLopt
using TSSOS
#using Clustering
using HDF5

#using Optim
using DifferentialEquations

###############################
#   SIMULATE DRIVED EVOLUTION
###############################


function sim_drv_lindblad(H0, V, J, ρ0, tₘₐₓ, dt, ω)

    L(ρ) = sum([Jᵢ * ρ * Jᵢ' - 0.5 * (Jᵢ' * Jᵢ * ρ + ρ * Jᵢ' * Jᵢ) for Jᵢ in J])
    Ht(t) = H0 + V * sin(ω * t)

    # Define the differential equation
    function matrix_ode!(dρ, ρ, p, t)
        dρ .= -im * (Ht(t) * ρ - ρ * Ht(t)) + L(ρ)
    end

    # Time span
    tspan = (0.0, tₘₐₓ)

    # Create an ODE problem
    prob = ODEProblem(matrix_ode!, ρ0, tspan)

    # Solve the ODE using a suitable solver, e.g., Tsit5 for non-stiff problems
    sol = solve(prob, dt=dt, adaptive = false)

    # Access solution
    ρ_solution = sol.u
    t_solution = sol.t;

    return(t_solution, ρ_solution)

end 


#########################
#   DRIVED SYSTEM DATA
#########################

function read_drived_evolution(file_name, freq_num)
    h5open(file_name, "r") do fid
        # Access a group named "group_name"
        group = fid[string(freq_num)]
        # Read a dataset within this group
        p0 = read(group["p0"])
        p1 = read(group["p1"])
        s_re = read(group["s_re"])
        s_im = read(group["s_im"])
        t = read(group["t"])

        n = length(t)
        ρ = [zeros(2, 2)+im*zeros(2, 2) for _ in 1:n]

        for i in [1:n;]
            ρ[i] = Hermitian([ p1[i]               s_re[i]+im*s_im[i]
                               s_re[i]-im*s_im[i]  p0[i]              ])
        end

        return t, ρ
    end
end

#######################################################
#  Kossakowski objective for drived two-level system  #
#######################################################

function kossak_cntrl_obj(ρ, t, H0ˢʸᵐᵇ, Vˢʸᵐᵇ, Cˢʸᵐᵇ, Fᴼᴺᴮ, ω)

    Hˢʸᵐᵇ(t) = H0ˢʸᵐᵇ + Vˢʸᵐᵇ * sin(ω * t)

    function Dc(ρ, t)
        U = (Hˢʸᵐᵇ(t)*ρ - ρ*Hˢʸᵐᵇ(t))/im 
        D = sum(Cˢʸᵐᵇ .* [2*fᵢ*ρ*fⱼ' - ρ*fⱼ'*fᵢ - fⱼ'*fᵢ*ρ  for fᵢ in Fᴼᴺᴮ, fⱼ in Fᴼᴺᴮ])/2
        return U + D
    end 

    obj = 0
    for i in 3:length(ρ)
        obj += LiPoSID.frobenius_norm2(
            ρ[i] - ρ[i-2] - (t[i]-t[i-1])*(Dc(ρ[i], t[i])+
            4*Dc(ρ[i-1], t[i-1])+Dc(ρ[i-2], t[i-2]))/3
        )
    end

    if isempty(monomials(obj))
        obj = 0. 
    else
        obj = sum(real(coef) * mon for (coef, mon) in zip(coefficients(obj), monomials(obj)))
    end

    return obj

end

#############################################################
#  Simulation of the drived evolution of two level sysstem  #
#  using Kossakowski parameters                             #
#############################################################
function kossak_cntrl_time_evolution(ρₒ, tₘₐₓ, dt, H0, V, C, Fᴼᴺᴮ, ω)

    D(ρ) = sum(C .* [2*fᵢ*ρ*fⱼ' - ρ*fⱼ'*fᵢ - fⱼ'*fᵢ*ρ  for fᵢ in Fᴼᴺᴮ, fⱼ in Fᴼᴺᴮ])/2
    Ht(t) = H0 + V * sin(ω * t)

    # Define the differential equation
    function matrix_ode!(dρ, ρ, p, t)
        dρ .= -im * (Ht(t) * ρ - ρ * Ht(t)) + D(ρ)
    end

    # Time span
    tspan = (0.0, tₘₐₓ)

    # Create an ODE problem
    prob = ODEProblem(matrix_ode!, ρₒ, tspan)

    # Solve the ODE using a suitable solver, e.g., Tsit5 for non-stiff problems
    sol = solve(prob, dt=dt, adaptive = false)

    # Access solution
    ρ_solution = sol.u
    t_solution = sol.t;
    
    return(t_solution, ρ_solution)

end 


######################################
# Constrained optimization with JuMP
######################################

# Helper function to suppress output
function suppress_output(f, args...; kwargs...)
    original_stdout = stdout
    original_stderr = stderr
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            return f(args...; kwargs...)
        end
    end
end

function jump(obj::Polynomial)
    try
        # Extract variables from the objective and constraints
        all_vars = variables(obj)
        n = length(all_vars)
        
        # Create the optimization model
        model = Model(Ipopt.Optimizer)
        
        # Define the JuMP variables
        @variable(model, x[1:n])
        
        # Create a function to evaluate the polynomial
        function evaluate_poly(poly, var_map)
            expr = zero(x[1]) # Initialize to zero with the type of x[1]
            for (term, coeff) in zip(terms(poly), coefficients(poly))
                term_expr = coeff
                for (v, exp) in zip(variables(term), exponents(term))
                    term_expr *= var_map[v]^exp
                end
                expr += term_expr
            end
            return expr
        end
        
        # Map the polynomial variables to JuMP variables
        var_map = Dict(v => x[i] for (i, v) in enumerate(all_vars))
        
        # Define the objective function
        obj_expr = evaluate_poly(obj, var_map)
        @NLobjective(model, Min, obj_expr)
                
        # Solve the optimization problem
        suppress_output(optimize!, model)
        
        # Get the results
        minimizer = value.(x)

        optimal_solution = all_vars => minimizer

        optimal_value = objective_value(model)
        status = termination_status(model)
        
        return optimal_solution, optimal_value, status
    catch e
        println("An error occurred: ", e)
        return nothing, nothing, :Error
    end
end


# Define the function jump_constrained
function jump_constrained(obj::Polynomial, constr::Vector{<:Polynomial})
    try
        # Extract variables from the objective and constraints
        obj_vars = variables(obj)
        constr_vars = unique(vcat(variables.(constr)...))
        all_vars = unique(vcat(obj_vars, constr_vars))
        n = length(all_vars)
        
        # Create the optimization model
        model = Model(Ipopt.Optimizer)
        
        # Define the JuMP variables
        @variable(model, x[1:n])
        
        # Create a function to evaluate the polynomial
        function evaluate_poly(poly, var_map)
            expr = zero(x[1]) # Initialize to zero with the type of x[1]
            for (term, coeff) in zip(terms(poly), coefficients(poly))
                term_expr = coeff
                for (v, exp) in zip(variables(term), exponents(term))
                    term_expr *= var_map[v]^exp
                end
                expr += term_expr
            end
            return expr
        end
        
        # Map the polynomial variables to JuMP variables
        var_map = Dict(v => x[i] for (i, v) in enumerate(all_vars))
        
        # Define the objective function
        obj_expr = evaluate_poly(obj, var_map)
        @NLobjective(model, Min, obj_expr)
        
        # Define the constraints
        for c in constr
            c_expr = evaluate_poly(c, var_map)
            @NLconstraint(model, c_expr >= 0)
        end
        
        # Solve the optimization problem
        suppress_output(optimize!, model)
        
        # Get the results
        minimizer = value.(x)

        optimal_solution = all_vars => minimizer

        optimal_value = objective_value(model)
        status = termination_status(model)
        
        return optimal_solution, optimal_value, status
    catch e
        println("An error occurred: ", e)
        return nothing, nothing, :Error
    end
end

function cs_tssos(obj::Polynomial, constr::Vector{<:Polynomial})
    ################################################################################################
    #
    #   Constrained TSSOS on polynomial without variable scaling
    #
    ################################################################################################

    # Collect all unique variables from the objective and constraints
    all_vars = union(variables(obj), reduce(union, variables.(constr)))

    # Initialize solution variables
    # solution = Dict(variables(obj) .=> nothing)
    solution = variables(obj) => nothing
    optimal_value = nothing
    status = :Error

    #relax_order = maxdegree(obj) > 2 ? maxdegree(obj) ÷ 2 : 2
    relax_order = maxdegree(obj) > 2 ? maxdegree(obj) : 2

    try 
        # Redirect stdout and stderr to suppress TSSOS output
        redirect_stdout(open("/dev/null", "w")) do
            redirect_stderr(open("/dev/null", "w")) do
                # Solve using cs_tssos_first
                opt, sol, data = TSSOS.cs_tssos_first([obj; constr...], all_vars, relax_order, numeq=0, solution=true, QUIET=true)

                previous_opt, previous_sol, previous_data = opt, sol, data 

                # Iteratively solve using tssos_higher! until no further solution is found
                while sol !== nothing
                    previous_opt, previous_sol, previous_data = opt, sol, data 
                    opt, sol, data = TSSOS.cs_tssos_higher!(data; QUIET=true, solution=true)
                end

                # Refine the solution
                ref_sol, flag = TSSOS.refine_sol(previous_opt, previous_sol, previous_data, QUIET=true, tol=1e-10)

                # Form the solution dictionary
                #solution = Dict(variables(obj) .=> ref_sol)
                solution = variables(obj) => ref_sol

                optimal_value = previous_opt
                status = flag == 0 ? :Global : :Local
            end
        end
    catch e
        println("TSSOS failed: ", e)
    end

    return solution, optimal_value, status
end

function tssos(obj::Polynomial)
    ################################################################################################
    #
    #    TSSOS on polynomial without variable scaling
    #
    ################################################################################################

    # Collect all unique variables from the objective and constraints
    all_vars = variables(obj)

    # Initialize solution variables
    # solution = Dict(variables(obj) .=> nothing)
    solution = variables(obj) => nothing
    optimal_value = nothing
    status = :Error

    #relax_order = maxdegree(obj) > 2 ? maxdegree(obj) ÷ 2 : 2
    relax_order = maxdegree(obj) > 2 ? maxdegree(obj) : 2

    try 
        # Redirect stdout and stderr to suppress TSSOS output
        redirect_stdout(open("/dev/null", "w")) do
            redirect_stderr(open("/dev/null", "w")) do
                # Solve using cs_tssos_first
                opt, sol, data = TSSOS.tssos_first(obj, all_vars, solution=true, QUIET=true)

                previous_opt, previous_sol, previous_data = opt, sol, data 

                # Iteratively solve using tssos_higher! until no further solution is found
                while sol !== nothing
                    previous_opt, previous_sol, previous_data = opt, sol, data 
                    opt, sol, data = TSSOS.tssos_higher!(data; QUIET=true, solution=true)
                end

                # Refine the solution
                ref_sol, flag = TSSOS.refine_sol(previous_opt, previous_sol, previous_data, QUIET=true, tol=1e-10)

                # Form the solution dictionary
                #solution = Dict(variables(obj) .=> ref_sol)
                solution = variables(obj) => ref_sol

                optimal_value = previous_opt
                status = flag == 0 ? :Global : :Local
            end
        end
    catch e
        println("TSSOS failed: ", e)
    end

    return solution, optimal_value, status
end

end