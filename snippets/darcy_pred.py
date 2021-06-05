from dolfin import *
import numpy

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

def primal(mesh, tag):
    "Standard H^1(mesh) formulation of Poisson's equation." 
    
    Q = FunctionSpace(mesh, "CG", 1)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    a = inner(grad(p), grad(q))*dx
    f = Constant(1.0)
    L = f*q*dx

    bc = DirichletBC(Q, 0.0, "on_boundary")

    A, b = assemble_system(a, L, bc)

    p = Function(Q)
    return (A, b, p)

def primal_lu(mesh, tag):
    "Solve primal H^1 formulation using LU."

    A, b, p = primal(mesh, tag)

    timer = Timer(tag)
    solver = LUSolver(A)
    solver.solve(p.vector(), b)
    timer.stop()

    return p

def primal_amg(mesh, tag):
    "Solve primal H^1 formulation using CG with AMG."

    A, b, p = primal(mesh, tag)

    timer = Timer(tag)
    solver = PETScKrylovSolver("cg", "amg")
    solver.set_operator(A)
    num_it = solver.solve(p.vector(), b)
    timer.stop()

    print("%s: num_it = " % tag, num_it)
    return p

def darcy(mesh):
    "Mixed H(div) x L^2 formulation of Poisson/Darcy."
    
    V = FiniteElement("RT", mesh.ufl_cell(), 1)
    Q = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, V*Q)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    a = (dot(u, v) + div(u)*q + div(v)*p)*dx
    f = Constant(-1.0)
    L = f*q*dx

    A = assemble(a)
    b = assemble(L)
    w = Function(W)

    return A, b, w
    
def darcy_lu(mesh, tag):
    "Solve mixed H(div) x L^2 formulation using LU"
    
    (A, b, w) = darcy(mesh)

    timer = Timer(tag)
    solve(A, w.vector(), b)
    timer.stop()

    return w

def darcy_prec1(W):
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    prec = (inner(u, v) + div(u)*div(v) + p*q)*dx
    B = assemble(prec)
    return B

def darcy_amg(mesh, tag):
    "Solve mixed H(div) x L^2 formulation using AMG"
    
    (A, b, w) = darcy(mesh)
    timer = Timer(tag)

    B = darcy_prec1(w.function_space())
    solver = PETScKrylovSolver("gmres", "amg")
    solver.set_operators(B,A)
    solver.parameters["monitor_convergence"] = True
    solver.parameters["report"] = True
    solver.parameters["relative_tolerance"] = 1.e-10
    num_iter = solver.solve(w.vector(), b)
    #solve(A, w.vector(), b)
    print("num_iter = ", num_iter)
    timer.stop()

    (u, p) = w.split(deepcopy=True)

    return w

def darcy_ilu(mesh, tag):
    "Solve mixed H(div) x L^2 formulation using iLU"
    
    (A, b, w) = darcy(mesh)
    timer = Timer(tag)
    solver = PETScKrylovSolver("gmres", "ilu")
    solver.set_operator(A)
    solver.parameters["monitor_convergence"] = True
    solver.parameters["report"] = True
    solver.parameters["relative_tolerance"] = 1.e-10
    num_iter = solver.solve(w.vector(), b)
    print("num_iter = ", num_iter)
    timer.stop()

    (u, p) = w.split(deepcopy=True)

    return w

def time_solve(mesh, algorithm, tag):

    solution = algorithm(mesh, tag)
    times = timings(TimingClear_clear, [TimingType_wall])
    dim = solution.function_space().dim()
    t = times.get_value(tag, "wall tot")

    return (t, dim)

def time_solves(mesh, algorithm, tag, R=1):

    times = numpy.empty(R)
    h = mesh.hmax()
    for i in range(R):
        t, dim = time_solve(mesh, algorithm, tag)
        print("%s (s) with N=%d and h=%.2g: %.3g" % (tag, dim, h, t))
        times[i] = t

    avg_t = numpy.mean(times)
    std_t = numpy.std(times)

    return (avg_t, std_t)

    
n = 16
mesh = UnitCubeMesh(n, n, n)
h = mesh.hmax()

sol = darcy_amg(mesh, "kk")