# test.jl

# Include the main code file
include("main.jl")  # Ensure that your main code is saved as main.jl

# Define a PauliAccess instance
coeffs = [1.0 + 0im, -0.5 + 0im]
paulis = ["XX", "ZZ"]  # Pauli strings for 2 qubits
n_qubits = 2

pauli_access = PauliAccess(coeffs, paulis, n_qubits)

# Define a Monomial function A^2
m = 2
monomial = Monomial(pauli_access, m)

# Define a MatrixElementProblem with indices and thresholds
i = 1
j = 1
ε = 0.01
g = 0.5
problem = MatrixElementProblem(monomial, i, j, ε, g)

# Solve the MatrixElementProblem
try
    result = solve(problem)
    println("Result of MatrixElementProblem: ", result)
catch e
    if isa(e, PromiseGapException)
        println("Promise gap encountered: ", e.message)
    else
        rethrow(e)
    end
end

# Define a TimeEvolution function with time t
t = 1.0
time_evolution = TimeEvolution(pauli_access, t)

# Define a LocalMeasurementProblem with thresholds
ε_local = 0.01
g_local = 0.5
local_problem = LocalMeasurementProblem(time_evolution, ε_local, g_local)

# Solve the LocalMeasurementProblem
try
    local_result = solve(local_problem)
    println("Result of LocalMeasurementProblem: ", local_result)
catch e
    if isa(e, PromiseGapException)
        println("Promise gap encountered: ", e.message)
    else
        rethrow(e)
    end
end

# Additional Test with SparseAccess
# Create a sparse matrix with Complex{Float64} entries
N_sparse = 4
s = 2  # Maximum nonzeros per column
# Example sparse diagonal matrix with Complex{Float64} entries
sparse_matrix = spdiagm(0 => [1.0+0im, 2.0+0im, 3.0+0im, 4.0+0im])

# Ensure the matrix meets the sparsity condition
sparse_access = SparseAccess(sparse_matrix, s)

# Define a Monomial function A^3
m_sparse = 3
monomial_sparse = Monomial(sparse_access, m_sparse)

# Define and solve a MatrixElementProblem
problem_sparse = MatrixElementProblem(monomial_sparse, 1, 1, 0.01, 5.0)
try
    result_sparse = solve(problem_sparse)
    println("Result with SparseAccess: ", result_sparse)
catch e
    if isa(e, PromiseGapException)
        println("Promise gap encountered: ", e.message)
    else
        rethrow(e)
    end
end
