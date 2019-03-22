from firedrake import *
#Citations.print_at_exit()
import numpy as np
from time import time
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Make mesh ghosted for evaluation of DG terms
parameters["ghost_mode"] = "shared_facet"

# Specifying function space polynomial degree, degree of integration rule, choice of benchmark solution and operator, number of mesh refinements, nature of boundary condition, choice of domain.
deg = 2
quad_deg = 8
prob = 0
meshrefs = 6
inhomogeneous = False
domain = 0

# Definining minimum function via conditional statements
def makemin(a,b):
    mkmin = conditional(le(a,b),a,0)+conditional(le(b,a),b,0)
    return mkmin
def model(quad_deg,deg):
    e_L2 = []; e_H1 = []; e_H2 = []; e_h1 = []; ndof = [];
    hm = []; muvec = []; etavec = []; e_inf = []; e_projerr = []; tt = [];
    for idx in range(meshrefs):
        if domain == 0:
            mesh = Mesh("Meshes/quasiunifrefdisk_%i.msh" % idx)
            Curved = True
        elif domain == 1:
            mesh = Mesh("Meshes/keyhole_%i.msh" % idx)
            Curved = True
        else:
            mesh = UnitSquareMesh(2**(idx+1),2**(idx+1))
            bdry_indicator = 0.0
            Curved = False
        # Implementing quadratic domain approximation if the domain has a curved boundary portion. Note, this currently applies to  the unit disk, and the "key-hole" shaped domain.
        if Curved == True:
            V = FunctionSpace(mesh, "CG", 2)
            bdry_indicator = Function(V)
            bc = DirichletBC(V, Constant(1.0), 1)
            bc.apply(bdry_indicator)
            VV = VectorFunctionSpace(mesh, "CG", 2)
            T = Function(VV)
            T.interpolate(SpatialCoordinate(mesh))
            # Quadratic interpolation of the unit disk chart x/|x|
            T.interpolate(conditional(abs(1-bdry_indicator) < 1e-5, T/sqrt(inner(T,T)), T))
            mesh = Mesh(T)
        else:
            mesh = UnitSquareMesh(2**(idx+1),2**(idx+1))
            bdry_indicator = 0.0
        # unit normal
        n = FacetNormal(mesh)

        # finite element space
        FES = FunctionSpace(mesh,"DG",deg)

        # Functions for defining bilinear form
        U = TrialFunction(FES)
        v = TestFunction(FES)

        # defining maximum mesh size
        DG = FunctionSpace(mesh, "DG", 0);
        h = sqrt(max(interpolate(CellVolume(mesh), DG).vector()[:]))

        # defining local mesh size
        hfct = sqrt(CellVolume(mesh))

        # defining jump stabilisation parameters
        eta1 = 10.0*pow(deg,4.0)

        x, y = SpatialCoordinate(mesh)

        # defining true solution, coefficient matrix A, and boundary condition function for benchmarks
        if prob == 0:
            rho = pow(x,2)+pow(y,2)
            d2udyy = Constant(1.0)
            u = 0.25*sin(pi*rho)
            trueu = u
            du0 = 0.5*pi*x*cos(pi*rho)
            du1 = 0.5*pi*y*cos(pi*rho)
            d2udxx = 0.5*pi*cos(pi*rho)-pi**2*x**2*sin(pi*rho)
            d2udyy = 0.5*pi*cos(pi*rho)-pi**2*y**2*sin(pi*rho)
            d2udxy = -pi**2*x*y*sin(pi*rho)
            f = -(d2udxx+d2udyy)
            g = u
            dg0 = du0
            dg1 = du1
            d2g00 = d2udxx
            d2g01 = d2udxy
            d2g11 = d2udyy
        
        elif prob == 1:
            u = sin(pi*x)*sin(pi*y)
            du0 = pi*cos(pi*x)*sin(pi*y)
            du1 = pi*sin(pi*x)*cos(pi*y)
            d2udxx = -pi*pi*u
            d2udyy = -pi*pi*u
            d2udxy = pi*pi*cos(pi*x)*cos(pi*y)
            f = -(d2udxx+d2udyy)

        # defining nondivergence part of the bilinear form
        def a(u,v):
            a = inner(grad(u),grad(v))*dx(mesh,degree = quad_deg)
            return a
        def B1in(u,v):
            B = -inner(avg(grad(u)),n('+'))*jump(v)*dS(mesh,degree = quad_deg)
            return B
        def B1out(u,v):
            B = -(inner(grad(u),n))*v*ds(mesh,degree = quad_deg)
            return B
        def B(u,v):
            B1 = B1out(u,v)+B1in(u,v)
            return B1
        def Jin(u,v):
            J1in = (eta1/h)*jump(u)*jump(v)*dS(mesh,degree = quad_deg)
            return J1in
        def Jout(u,v):
            J1out = (eta1/h)*u*v*ds(mesh,degree = quad_deg)
            return J1out
        def J(u,v):
            J1 = Jin(u,v)+Jout(u,v)
            return J1
        
        # defining bilinear form
        A = a(U,v)\
            +B(U,v)\
            +B(v,U)\
            +J(U,v)
        # defining linear form
        L = (f)*(v)*dx(mesh,degree = quad_deg)

        # defining solution function
        U = Function(FES)

        # begin timing of linear system solve
        t = time()

        # solving linear system
        solve(A == L, U,
              solver_parameters = {
              "ksp_type": "preonly",
              "pc_type": "lu"})
        # end timing of linear system solve
        tt.append(time()-t)
        hm.append(h)

        # calculating errors
        errorL2 = (U-u)**2*dx(mesh,degree = quad_deg)
        errorH1 = ((U-u).dx(0)**2+(U-u).dx(1)**2)*dx(mesh,degree = quad_deg)+J(U-u,U-u)
        eL2 = sqrt(assemble(errorL2))
        eH1 = sqrt(assemble(errorH1))
        e_L2.append(eL2)
        e_H1.append(eH1)

        # obtain number of DoFs
        ndof.append(U.vector().array().size)

    EOCL2 = []
    EOCH1 = []
    EOCH2 = []
    EOCh1 = []
    EOClinf = []
    EOCl2proj = []


    EOCL2.append(0)
    EOCH1.append(0)
    EOClinf.append(0)
    EOCl2proj.append(0)

    #Calcuating error orders of convergence.
    for k in range(1,len(e_L2)):
        EOCL2.append(ln(e_L2[k-1]/(e_L2[k]))/ln(hm[k-1]/hm[k]))
        EOCH1.append(ln(e_H1[k-1]/(e_H1[k]))/ln(hm[k-1]/hm[k]))
    k = 0

    # outputting jump stabilisation constants
    print("etacon = ",eta1)

    # outputting error results
    for k in range(len(e_L2)):
        print( "NDoFs = ", ndof[k])
        print( "Mesh size = ", hm[k])
        print( "run time = ",tt[k])
        print( "||u - u_h||_0 = ", e_L2[k], "   EOC = ", EOCL2[k])
        print( "||u - u_h||_1 = ", e_H1[k], "   EOC = ", EOCH1[k])
        k = k+1
    # returning errors, NDofs, runtimes, EOCs and polynomial degree for the creation of data files
    return e_L2, e_H1, ndof, hm, tt,  EOCL2, EOCH1, deg
# solving problem for degrees 2, 3 and 4, and saving data to text file
for degg in [1,2,3]:
    e_L2, e_H1,ndof, hm, tt,  EOCL2, EOCH1, deg = model(10,degg)
    print( "polynomial degree: ", deg)
    print( "integral degree: ", quad_deg)
    print( "experiment no. : ", prob)
    out_name1 = "Curved-poisson/e_L2p%i.txt" %deg
    out_name2 = "Curved-poisson/e_H1p%i.txt" %deg
    out_name5 = "Curved-poisson/EOCL2p%i.txt" %deg
    out_name6 = "Curved-poisson/EOCH1p%i.txt" %deg
    out_name9 = "Curved-poisson/dofsp%i.txt" %deg
    out_name10 = "Curved-poisson/meshsize.txt"
    out_name11 = "Curved-poisson/timesp%i.txt" %deg


    np.savetxt(out_name1,e_L2,fmt = '%s')
    np.savetxt(out_name2,e_H1,fmt = '%s')
    np.savetxt(out_name5,EOCL2,fmt = '%s')
    np.savetxt(out_name6,EOCH1,fmt = '%s')
    np.savetxt(out_name9,ndof,fmt = '%s')
    np.savetxt(out_name10,hm,fmt = '%s')
    np.savetxt(out_name11,tt,fmt = '%s')
