
from dolfin import *
from dolfin_adjoint import *
import time
import sys

import itertools

def getPermutedListOfFour():
    initialPermutation = itertools.product(range(2), repeat=2)
    resultList = [i for i in [*initialPermutation] if sum(i) > 0]
    # print(resultList)
    return resultList

def getBCs():
    in_BCs, out_BCs = getPermutedListOfFour(), getPermutedListOfFour()
    return (in_BCs, out_BCs) # up and sides

try:
    from pyadjoint import ipopt  # noqa: F401
except ImportError:
    print("""This example depends on IPOPT and Python ipopt bindings. \
  When compiling IPOPT, make sure to link against HSL, as it \
  is a necessity for practical problems.""")
    raise

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

mu = Constant(1.0)                   # viscosity
alphaunderbar = 2.5 * mu / (100**2)  # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discrete-valuedness of solution

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

# Next we define the mesh (a rectangle 1 high and :math:`\delta` wide)
# and the function spaces to be used for the control :math:`\rho`, the
# velocity :math:`u` and the pressure :math:`p`. Here we will use the
# Taylor-Hood finite element to discretise the Stokes equations
# :cite:`taylor1973`.

N = 40
delta = 1.  # The aspect ratio of the domain, 1 high and \delta wide

mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), N, N))
A = FunctionSpace(mesh, "CG", 1)        # control function space

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

in_BC, out_BC = getBCs()
# in_BC_str = [str(i) for i in in_BC]
# out_BC_str = [str(i) for i in out_BC]

for in_permutation in in_BC:
    for out_permutation in out_BC:

        in_permutation_str = [str(i) for i in in_permutation]
        out_permutation_str = [str(i) for i in out_permutation]
        # V = Constant(1.0/3) * delta  # want the fluid to occupy 1/3 of the domain

        # print(f"intake: {intakepipes}")
        # print(f"outtake: {outtakepipes}")

        class InflowOutflow(UserExpression):
            def eval(self, values, x):
                inputMultiplier = 1 / sum(in_permutation)
                outputMultiplier = 1 / sum(out_permutation)
                values[1] = 0.0
                values[0] = 0.0
                l = 1.0/6.0
                gbar = 1.0
        # inflow boundary conditions horizontal
                if x[0] == 0.0:
                    if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2) and in_permutation[0] == 1:
                        t = x[1] - 1.0/4
                        values[0] = inputMultiplier * gbar * (1 - (2*t/l)**2)

                    if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2) and in_permutation[1] == 1:
                        t = x[1] - 3.0/4
                        values[0] = inputMultiplier * gbar * (1 - (2*t/l)**2)
        # outflow boundary conditions horizontal
                if x[0] == delta:
                    if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2) and out_permutation[0] == 1:
                        t = x[1] - 1.0/4
                        values[0] = outputMultiplier * gbar * (1 - (2*t/l)**2)

                    if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2) and out_permutation[1] == 1:
                        t = x[1] - 3.0/4
                        values[0] = outputMultiplier * gbar * (1 - (2*t/l)**2)
            
            def value_shape(self):
                return (2,)
            
        def forward(rho):
            """Solve the forward problem for a given fluid distribution rho(x)."""
            w = Function(W)
            (u, p) = TrialFunctions(W)
            (v, q) = TestFunctions(W)

            F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
                inner(grad(p), v) * dx  + inner(div(u), q) * dx)
            bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
            solve(lhs(F) == rhs(F), w, bcs=bc)

            return w
        
        # main loop
        for i in range(10,45):
            V = Constant(i/100) * delta  # want the fluid to occupy varied fraction of the domain

            rho = interpolate(Constant(float(V)/delta), A)
            w = forward(rho)
            (u, p) = split(w)


            controls = File(f"fourpipe/output/input-{'-'.join(in_permutation_str)}/output-{'-'.join(out_permutation_str)}/control_iterations_guess.pvd")
            allctrls = File(f"fourpipe/output/input-{'-'.join(in_permutation_str)}/output-{'-'.join(out_permutation_str)}/allcontrols.pvd")
            rho_viz = Function(A, name="ControlVisualisation")
            def eval_cb(j, rho):
                rho_viz.assign(rho)
                controls << rho_viz
                allctrls << rho_viz
                with open(f"fourpipe/callbacks/JValueTest_input-{'-'.join(in_permutation_str)}_output-{'-'.join(out_permutation_str)}_frac{i}.txt", "w+") as f:
                    f.write(str(j))
                with open(f'./dataforpaper/regularcallback/modeltest_in-{in_permutation[0]}-{in_permutation[1]}_out-{out_permutation[0]}-{out_permutation[1]}_{i}.txt', 'w+') as f:
                    f.write(f'Objective function value: {j}\n')

            J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
            m = Control(rho)
            Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)



            # # Bound constraints
            lb = 0.0
            ub = 1.0

            volume_constraint = UFLInequalityConstraint((V/delta - rho)*dx, m)

            # Solve the optimization problem with q = 0.01
            problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
            parameters = {'maximum_iterations': 20} #20

            solver = IPOPTSolver(problem, parameters=parameters)
            start1 = time.perf_counter()
            rho_opt = solver.solve()
            elapsed1 = time.perf_counter() - start1

            rho_opt_xdmf = XDMFFile(MPI.comm_world, f"fourpipe/output/input-{'-'.join(in_permutation_str)}/output-{'-'.join(out_permutation_str)}/control_solution_guess.xdmf")
            rho_opt_xdmf.write(rho_opt)



            q.assign(0.1)
            rho.assign(rho_opt)
            set_working_tape(Tape())

            rho_intrm = XDMFFile(MPI.comm_world, "intermediate-guess-%s.xdmf" % N)
            rho_intrm.write(rho)

            w = forward(rho)
            (u, p) = split(w)

            # Define the reduced functionals
            controls = File(f"fourpipe/controloutput/input-{'-'.join(in_permutation_str)}/output-{'-'.join(out_permutation_str)}/fraction{i}/control_iterations_final.pvd")
            rho_viz = Function(A, name="ControlVisualisation")
            def eval_cb(j, rho):
                rho_viz.assign(rho)
                controls << rho_viz
                allctrls << rho_viz

            J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
            m = Control(rho)
            Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

            problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
            parameters = {'maximum_iterations': 30} #30

            solver = IPOPTSolver(problem, parameters=parameters)

            start_time = time.perf_counter()
            rho_opt = solver.solve()
            elapsed_time = time.perf_counter() - start_time
            with open(f'./dataforpaper/regularexectime/modeltest_in-{in_permutation[0]}-{in_permutation[1]}_out-{out_permutation[0]}-{out_permutation[1]}_{i}.txt', 'w+') as f:
                    f.write(f'Elapsed time (s): {elapsed_time + elapsed1}')

            
            print(type(rho_opt))
            print(rho_opt)
            
            rho_opt_final = XDMFFile(MPI.comm_world, f"fourpipe/finaloutputs/input-{'-'.join(in_permutation_str)}/output-{'-'.join(out_permutation_str)}/control_solution_final_{i}.xdmf")
            rho_opt_final.write(rho_opt)

    