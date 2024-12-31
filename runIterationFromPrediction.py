from dolfin import *
from dolfin_adjoint import *
import numpy as np
import os
import time

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

N = 40
delta = 1.  # The aspect ratio of the domain, 1 high and \delta wide

# pred: numpy array (needs to be flattened)

def numpyToDolfinMesh(array):

    # Create the RectangleMesh
    mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), N, N))
    V = FunctionSpace(mesh, "CG", 1)
    dofmap = V.dofmap()

    # Get the coordinates of the mesh vertices (all vertex coordinates)
    vertices = mesh.coordinates()

    # Assign to Dolfin Function
    u = Function(V)
    # u.vector()[:] = values

    # # Flatten the NumPy array
    # np_array_flat = array.flatten()

    # Loop through all vertices and assign values from the NumPy array to the corresponding dof
    for i, vertex in enumerate(vertices):
        # Get the corresponding value from the NumPy array
        x, y = vertex
        row = int(x * (array.shape[0] - 1))
        col = int(y * (array.shape[1] - 1))
        value = array[col, row]

        # Get the global dof index corresponding to the vertex
        dof_index = dofmap.entity_dofs(mesh, 0)[i]

        # Assign the value to the function (using the dof index)
        u.vector()[dof_index] = value
    
    return u

# samplePred = np.load('./neuralnetworkdata/TopologyOutput/input-0-1_output-0-1_29.npy')  # Replace with your data

def run_additional_iteration(pred, inflow_1, inflow_2, outflow_1, outflow_2, volfrac):
    mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), N, N))
    A = FunctionSpace(mesh, "CG", 1)        # control function space

   

    class InflowOutflow(UserExpression):
        def eval(self, values, x):
            inputMultiplier = 1 / (inflow_1 + inflow_2)
            outputMultiplier = 1 / (outflow_1 + outflow_2)
            values[1] = 0.0
            values[0] = 0.0
            l = 1.0/6.0
            gbar = 1.0
    # inflow boundary conditions horizontal
            if x[0] == 0.0:
                if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2) and inflow_1 == 1:
                    t = x[1] - 1.0/4
                    values[0] = inputMultiplier * gbar * (1 - (2*t/l)**2)

                if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2) and inflow_2 == 1:
                    t = x[1] - 3.0/4
                    values[0] = inputMultiplier * gbar * (1 - (2*t/l)**2)
    # outflow boundary conditions horizontal
            if x[0] == delta:
                if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2) and outflow_1 == 1:
                    t = x[1] - 1.0/4
                    values[0] = outputMultiplier * gbar * (1 - (2*t/l)**2)

                if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2) and outflow_2 == 1:
                    t = x[1] - 3.0/4
                    values[0] = outputMultiplier * gbar * (1 - (2*t/l)**2)
        
        def value_shape(self):
            return (2,)
        
    def forward(rho):
        """Solve the forward problem for predicted fluid distribution rho(x)."""
        w = Function(W)
        # add the predicted values from the neural network
        # w.vector()[:] = pred

        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)

        F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
            inner(grad(p), v) * dx  + inner(div(u), q) * dx)
        bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
        solve(lhs(F) == rhs(F), w, bcs=bc)

        return w
    
    V = Constant(volfrac/100) * delta  # want the fluid to occupy varied fraction of the domain
    

    rho = interpolate(Constant(float(V)/delta), A)


    controls = File(f"iterationsfrompredicted/output/input-{inflow_1}-{inflow_2}/output-{outflow_1}-{outflow_2}/control_iterations_guess.pvd")
    allctrls = File(f"iterationsfrompredicted/output/input-{inflow_1}-{inflow_2}/output-{outflow_1}-{outflow_2}/allcontrols.pvd")
    rho_viz = Function(A, name="ControlVisualisation")

    # # Bound constraints
    lb = 0.0
    ub = 1.0

    U_h = VectorElement("CG", mesh.ufl_cell(), 2)
    P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, U_h*P_h) 

    q.assign(0.1)
    rho.assign(numpyToDolfinMesh(pred))
    set_working_tape(Tape())

    rho_intrm = XDMFFile(MPI.comm_world, "intermediate-guess-%s.xdmf" % N)
    rho_intrm.write(rho)

    w = forward(rho)
    (u, p) = split(w)

    # Define the reduced functionals
    controls = File(f"iterationsfrompredicted/controloutput/input-{inflow_1}-{inflow_2}/output-{outflow_1}-{outflow_2}/fraction{volfrac}/control_iterations_final.pvd")
    rho_viz = Function(A, name="ControlVisualisation")
    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz
        with open(f'./dataforpaper/iterationsfrompredictioncallback/modeltest_in-{inflow_1}-{inflow_2}_out-{outflow_1}-{outflow_2}_{volfrac}.txt', 'w+') as f:
            f.write(f'Objective function value: {j}\n')

    J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
    m = Control(rho)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

    volume_constraint = UFLInequalityConstraint((V/delta - rho)*dx, m)

# We can now solve the optimisation problem with :math:`q=0.1`, starting
# from the solution of :math:`q=0.01`:
    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    parameters = {'maximum_iterations': 5} 

    solver = IPOPTSolver(problem, parameters=parameters)

    start_time = time.perf_counter()
    rho_opt = solver.solve()
    elapsed_time = time.perf_counter() - start_time
    
    print(type(rho_opt))
    print(rho_opt)
    
    rho_opt_final = XDMFFile(MPI.comm_world, f"iterationsfrompredicted/finaloutputs/input-{inflow_1}-{inflow_2}/output-{outflow_1}-{outflow_2}/control_solution_final_{volfrac}.xdmf")
    rho_opt_final.write(rho_opt)

    return rho_opt, elapsed_time

predictedTopologyDir = './modelnumpyoutputs/'

topologies = os.listdir(predictedTopologyDir)

for filename in topologies:
    topologyData = filename.split('_')
    inflow = [int(i) for i in topologyData[1].split('-')[1:]]
    outflow = [int(i) for i in topologyData[2].split('-')[1:]]
    volfrac = int(topologyData[3].split('.')[0])
    pred = np.load(predictedTopologyDir + filename)

    print(inflow)
    print(outflow)

    rho_opt_final, timeElapsed = run_additional_iteration(pred, inflow[0], inflow[1], outflow[0], outflow[1], volfrac)


    print(f'Elapsed time (s): {timeElapsed}')
    with open(f'./dataforpaper/feaexectime/modeltest_in-{inflow[0]}-{inflow[1]}_out-{outflow[0]}-{outflow[1]}_{volfrac}.txt', 'w+') as f:
        f.write(f'Elapsed time (s): {timeElapsed}')

# for inBC in inPossibilities:
#     for outBC in outPossibilities:
#         for volfrac in range(10, 50):
#             start_time = time.perf_counter()
#             modelInput = torch.tensor(np.array([[bc[0]], [bc[1]], np.full((1, 41), volfrac)])).to(device).to(torch.float32)
#             output = model(modelInput)
#             output = output.cpu().detach().numpy()
#             output = np.reshape(output, (41, 41))
#             elapsed_time = time.perf_counter() - start_time


# run_additional_iteration(samplePred, 0, 1, 0, 1, 29)