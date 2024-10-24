using LinearAlgebra
using SparseArrays

# Abstract type for matrix access
abstract type MatrixAccess end

# Sparse matrix access
struct SparseAccess <: MatrixAccess
    matrix::SparseMatrixCSC{ComplexF64}
    s::Int  # sparsity parameter (max nonzeros per column)
    N::Int  # dimension

    function SparseAccess(matrix::SparseMatrixCSC{ComplexF64}, s::Int)
        N = size(matrix, 1)
        # Verify sparsity condition
        max_nonzeros_per_column = maximum(diff(matrix.colptr))
        @assert max_nonzeros_per_column <= s "Matrix exceeds specified sparsity s"
        new(matrix, s, N)
    end
end

struct PauliAccess <: MatrixAccess
    coeffs::Vector{ComplexF64} 
    paulis::Vector{String}      
    N::Int                      
    L::Int                     

    function PauliAccess(coeffs::Vector{ComplexF64}, paulis::Vector{String}, n_qubits::Int)
        N = 2^n_qubits
        L = length(coeffs)
        @assert length(paulis) == L "Number of coefficients must match number of Pauli strings"
        @assert all(length.(paulis) .== n_qubits) "All Pauli strings must have length n_qubits"
        @assert all(all(c in "IXYZ" for c in p) for p in paulis) "Invalid Pauli characters"
        new(coeffs, paulis, N, L)
    end
end

# Abstract type for matrix functions
abstract type MatrixFunction end

# Matrix monomial A^m
mutable struct Monomial <: MatrixFunction
    access::MatrixAccess
    m::Int
    matrix_cache::Union{Nothing, Matrix{ComplexF64}}

    function Monomial(access::MatrixAccess, m::Int)
        new(access, m, nothing)
    end
end

# Chebyshev polynomial T_m(A)
mutable struct ChebyshevPolynomial <: MatrixFunction
    access::MatrixAccess
    m::Int
    matrix_cache::Union{Nothing, Matrix{ComplexF64}}

    function ChebyshevPolynomial(access::MatrixAccess, m::Int)
        new(access, m, nothing)
    end
end

# Time evolution operator e^{-iAt}
mutable struct TimeEvolution <: MatrixFunction
    access::MatrixAccess
    t::Float64
    matrix_cache::Union{Nothing, Matrix{ComplexF64}}

    function TimeEvolution(access::MatrixAccess, t::Float64)
        new(access, t, nothing)
    end
end

# Matrix inverse function
mutable struct MatrixInverse <: MatrixFunction
    access::MatrixAccess
    κ::Float64  # condition number
    matrix_cache::Union{Nothing, Matrix{ComplexF64}}

    function MatrixInverse(access::MatrixAccess, κ::Float64)
        new(access, κ, nothing)
    end
end

# Problem definitions
struct MatrixElementProblem
    func::MatrixFunction
    i::Int      # row index
    j::Int      # column index
    ε::Float64  # precision
    g::Float64  # threshold
end

struct LocalMeasurementProblem
    func::MatrixFunction
    ε::Float64  # precision
    g::Float64  # threshold
end

# Pauli matrix constants
const σI = [1.0 + 0im 0.0 + 0im; 0.0 + 0im 1.0 + 0im]
const σX = [0.0 + 0im 1.0 + 0im; 1.0 + 0im 0.0 + 0im]
const σY = [0.0 + 0im -1im; 1im 0.0 + 0im]
const σZ = [1.0 + 0im 0.0 + 0im; 0.0 + 0im -1.0 + 0im]
const PAULI_DICT = Dict('I'=>σI, 'X'=>σX, 'Y'=>σY, 'Z'=>σZ)

# Convert Pauli string to matrix
function pauli_to_matrix(pauli_string::String)
    pauli_chars = collect(pauli_string)
    result = PAULI_DICT[pauli_chars[1]]
    for p in pauli_chars[2:end]
        result = kron(result, PAULI_DICT[p])
    end
    return result
end

# Evaluate matrix functions
function evaluate(M::Monomial)
    if M.matrix_cache === nothing
        if isa(M.access, SparseAccess)
            M.matrix_cache = Matrix(M.access.matrix)  # Convert SparseMatrixCSC to Dense Matrix
        else  # PauliAccess
            matrix = zeros(ComplexF64, M.access.N, M.access.N)
            for (c, p) in zip(M.access.coeffs, M.access.paulis)
                matrix += c * pauli_to_matrix(p)
            end
            M.matrix_cache = matrix
        end
    end
    return M.matrix_cache^M.m
end

function evaluate(C::ChebyshevPolynomial)
    if C.matrix_cache === nothing
        # Construct matrix A
        if isa(C.access, SparseAccess)
            A = Matrix(C.access.matrix)  # Convert SparseMatrixCSC to Dense Matrix
        else  # PauliAccess
            A = zeros(ComplexF64, C.access.N, C.access.N)
            for (c, p) in zip(C.access.coeffs, C.access.paulis)
                matrix += c * pauli_to_matrix(p)
            end
        end
        m = C.m
        N = C.access.N
        # Compute Chebyshev polynomial T_m(A)
        if m == 0
            C.matrix_cache = Matrix{ComplexF64}(I, N, N)
        elseif m == 1
            C.matrix_cache = A
        else
            T_prev = Matrix{ComplexF64}(I, N, N)
            T_curr = A
            for _ in 2:m
                T_next = 2 * A * T_curr - T_prev
                T_prev = T_curr
                T_curr = T_next
            end
            C.matrix_cache = T_curr
        end
    end
    return C.matrix_cache
end

function evaluate(T::TimeEvolution)
    if T.matrix_cache === nothing
        # Construct matrix A
        if isa(T.access, SparseAccess)
            A = Matrix(T.access.matrix)  # Convert SparseMatrixCSC to Dense Matrix
        else  # PauliAccess
            A = zeros(ComplexF64, T.access.N, T.access.N)
            for (c, p) in zip(T.access.coeffs, T.access.paulis)
                A += c * pauli_to_matrix(p)
            end
        end
        t = T.t
        T.matrix_cache = exp(-im * A * t)
    end
    return T.matrix_cache
end

function evaluate(MI::MatrixInverse)
    if MI.matrix_cache === nothing
        if isa(MI.access, SparseAccess)
            A = Matrix(MI.access.matrix)  
        else  # PauliAccess
            A = zeros(ComplexF64, MI.access.N, MI.access.N)
            for (c, p) in zip(MI.access.coeffs, MI.access.paulis)
                A += c * pauli_to_matrix(p)
            end
        end
        # Compute matrix inverse
        MI.matrix_cache = inv(A)
    end
    return MI.matrix_cache
end

struct PromiseGapException <: Exception
    message::String
end

function solve(problem::MatrixElementProblem)
    result = evaluate(problem.func)
    value = abs(result[problem.i, problem.j])

    if value >= problem.g + problem.ε
        return true
    elseif value <= problem.g - problem.ε
        return false
    else
        throw(PromiseGapException("Value falls in promise gap"))
    end
end

function solve(problem::LocalMeasurementProblem)
    matrix = evaluate(problem.func)
    N = size(matrix, 1)
    n_qubits = Int(round(log2(N)))  # Ensure n_qubits is an integer
    
    ψ₀ = zeros(ComplexF64, N)
    ψ₀[1] = 1.0
    
    ψ = matrix * ψ₀

    π_single_qubit = [1.0 + 0im 0.0 + 0im; 0.0 + 0im 0.0 + 0im]
    π = kron(π_single_qubit, Matrix{ComplexF64}(I, div(N, 2), div(N, 2)))
    # Compute measurement
    value = real(ψ' * π * ψ)

    if value >= problem.g + problem.ε
        return true
    elseif value <= problem.g - problem.ε
        return false
    else
        throw(PromiseGapException("Value falls in promise gap"))
    end
end
