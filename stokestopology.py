
from dolfin import *
from dolfin_adjoint import *
import time
import sys

# Next we import the Python interface to IPOPT. If IPOPT is
# unavailable on your system, we strongly :doc:`suggest you install it
# <../../download/index>`; IPOPT is a well-established open-source
# optimisation algorithm.

import itertools

try:
    from pyadjoint import ipopt  # noqa: F401
except ImportError:
    print("""This example depends on IPOPT and Python ipopt bindings. \
  When compiling IPOPT, make sure to link against HSL, as it \
  is a necessity for practical problems.""")
    raise

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

# Next we define some constants, and define the inverse permeability as
# a function of :math:`\rho`.

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

N = 35
delta = 1.  # The aspect ratio of the domain, 1 high and \delta wide

mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), N, N))
A = FunctionSpace(mesh, "CG", 1)        # control function space

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

# Define the boundary condition on velocity

# class InflowOutflow(UserExpression):
#     def eval(self, values, x):
#         values[1] = 0.0
#         values[0] = 0.0
#         l = 1.0/6.0
#         gbar = 1.0

# # outward boundary conditions
#         if x[0] == 0.0:
#             if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
#                 t = x[1] - 1.0/4
#                 values[0] = gbar*(1 - (2*t/l)**2)
#             if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
#                 t = x[1] - 3.0/4
#                 values[0] = gbar*(1 - (2*t/l)**2)
# # inner boundary conditions
#         if x[0] == delta:
#             if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
#                 t = x[1] - 1.0/4
#                 values[0] = gbar*(1 - (2*t/l)**2)
#             if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
#                 t = x[1] - 3.0/4
#                 values[0] = gbar*(1 - (2*t/l)**2)

#     def value_shape(self):
#         return (2,)

# class InflowOutflow(UserExpression):
#     def eval(self, values, x):
#         values[1] = 0.0
#         values[0] = 0.0
#         l = 1.0/6.0
#         gbar = 1.0

# # inflow boundary conditions
#         if x[0] == 0.0:
#             if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
#                 t = x[1] - 1.0/4
#                 values[0] = gbar*(1 - (2*t/l)**2)
#             if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
#                 t = x[1] - 3.0/4
#                 values[0] = gbar*(1 - (2*t/l)**2)
# # outflow boundary conditions
#         if x[0] == delta:
#             # if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
#             #     t = x[1] - 1.0/4
#             #     values[0] = gbar*(1 - (2*t/l)**2)
#             # if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
#             #     t = x[1] - 3.0/4
#             #     values[0] = 2*gbar*(1 - (2*t/l)**2)
#             if (1.0/2 - l/2) < x[1] < (1.0/2 + l/2):
#                 t = x[1] - 1.0/2
#                 values[0] = 2*gbar*(1 - (2*t/l)**2)

#     def value_shape(self):
#         return (2,)
# Next we define a function that given a control :math:`\rho` solves the
# forward PDE for velocity and pressure :math:`(u, p)`. (The advantage
# of formulating it in this manner is that it makes it easy to conduct
# :doc:`Taylor remainder convergence tests
# <../../documentation/verification>`.)


# def forward(rho):
#     """Solve the forward problem for a given fluid distribution rho(x)."""
#     w = Function(W)
#     (u, p) = TrialFunctions(W)
#     (v, q) = TestFunctions(W)

#     F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
#          inner(grad(p), v) * dx  + inner(div(u), q) * dx)
#     bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
#     solve(lhs(F) == rhs(F), w, bcs=bc)

#     return w

# Now we define the ``__main__`` section. We define the initial guess
# for the control and use it to solve the forward PDE. In order to
# ensure feasibility of the initial control guess, we interpolate the
# volume bound; this ensures that the integral constraint and the bound
# constraint are satisfied.

for left in range(1, 4):
    for right in range(1,4):
        intakepipes = right
        outtakepipes = left

        # V = Constant(1.0/3) * delta  # want the fluid to occupy 1/3 of the domain

        print(f"intake: {intakepipes}")
        print(f"outtake: {outtakepipes}")

        class InflowOutflow(UserExpression):
            def eval(self, values, x):
                outputmultiplier = left / right
                leftMultiplier = 1 / left
                rightMultiplier = 1 / right
                values[1] = 0.0
                values[0] = 0.0
                l = 1.0/7.0
                gbar = 1.0
        # inflow boundary conditions horizontal
                if x[0] == 0.0:
                    # if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
                    #     t = x[1] - 1.0/4
                    #     values[0] = gbar*(1 - (2*t/l)**2)

                    if (3./14 - l/2) < x[1] < (3./14 + l/2):
                        t = x[1] - 3./14
                        values[0] = leftMultiplier * gbar*(1 - (2*t/l)**2)

                    if (left == 2 or left == 3) and (1./2 - l/2) < x[1] < (1./2 + l/2):
                        t = x[1] - 1./2
                        values[0] = leftMultiplier * gbar * (1 - (2*t/l)**2)
                    
                    if left == 3 and (11./14 - l/2) < x[1] < (11./14 + l/2):
                        t = x[1] - 11./14
                        values[0] = leftMultiplier * gbar * (1 - (2*t/l)**2)
        # outflow boundary conditions horizontal
                if x[0] == delta:
                    if (3./14 - l/2) < x[1] < (3./14 + l/2):
                        t = x[1] - 3./14
                        values[0] = rightMultiplier * gbar * (1 - (2*t/l)**2)

                    if (right == 2 or right == 3) and (1./2 - l/2) < x[1] < (1./2 + l/2):
                        t = x[1] - 1./2
                        values[0] = rightMultiplier * gbar * (1 - (2*t/l)**2)
                    
                    if right == 3 and (11./14 - l/2) < x[1] < (11./14 + l/2):
                        t = x[1] - 11./14
                        values[0] = rightMultiplier * gbar * (1 - (2*t/l)**2)
            
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
        for i in range(25,75):
            V = Constant((100 - i)/100) * delta  # want the fluid to occupy varied fraction of the domain

            rho = interpolate(Constant(float(V)/delta), A)
            w = forward(rho)
            (u, p) = split(w)


            controls = File(f"output/left{left}/right{right}/control_iterations_guess.pvd")
            allctrls = File(f"output/left{left}/right{right}/allcontrols.pvd")
            rho_viz = Function(A, name="ControlVisualisation")
            def eval_cb(j, rho):
                rho_viz.assign(rho)
                controls << rho_viz
                allctrls << rho_viz
                with open(f"callbacks/JValueTest_left{left}_right{right}_frac{i}.txt", "w+") as f:
                    f.write(str(j))

        # Now we define the functional and :doc:`reduced functional
        # <../maths/2-problem>`:

            J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
            m = Control(rho)
            Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)



            # # Bound constraints
            lb = 0.0
            ub = 1.0

            volume_constraint = UFLInequalityConstraint((V/delta - rho)*dx, m)

            # Solve the optimization problem with q = 0.01
            problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
            parameters = {'maximum_iterations': 10} #20

            solver = IPOPTSolver(problem, parameters=parameters)
            rho_opt = solver.solve()

            rho_opt_xdmf = XDMFFile(MPI.comm_world, f"output/left{left}/right{right}/control_solution_guess.xdmf")
            rho_opt_xdmf.write(rho_opt)



            q.assign(0.1)
            rho.assign(rho_opt)
            set_working_tape(Tape())

            rho_intrm = XDMFFile(MPI.comm_world, "intermediate-guess-%s.xdmf" % N)
            rho_intrm.write(rho)

            w = forward(rho)
            (u, p) = split(w)

            # Define the reduced functionals
            controls = File(f"controloutput/left{left}/right{right}/fraction{i}/control_iterations_final.pvd")
            rho_viz = Function(A, name="ControlVisualisation")
            def eval_cb(j, rho):
                rho_viz.assign(rho)
                controls << rho_viz
                allctrls << rho_viz

            J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
            m = Control(rho)
            Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

        # We can now solve the optimisation problem with :math:`q=0.1`, starting
        # from the solution of :math:`q=0.01`:
            problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
            parameters = {'maximum_iterations': 20} #30

            solver = IPOPTSolver(problem, parameters=parameters)
            rho_opt = solver.solve()

            
            print(type(rho_opt))
            print(rho_opt)
            
            rho_opt_final = XDMFFile(MPI.comm_world, f"finaloutputs/left{left}/right{right}/control_solution_final_{i}.xdmf")
            rho_opt_final.write(rho_opt)

    