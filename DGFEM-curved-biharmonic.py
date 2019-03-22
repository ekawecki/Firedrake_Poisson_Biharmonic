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
SM = False
# Definining minimum function via conditional statements
def makemin(a,b):
    mkmin = conditional(le(a,b),a,0)+conditional(le(b,a),b,0)
    return mkmin
def model(quad_deg,deg):
    e_L2 = []; e_H1 = []; e_H2 = []; e_EN = []; ndof = [];
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
        hin = makemin(hfct('+'),hfct('-'))
        hout = hfct

        # defining jump stabilisation parameters
        if deg == 2:
            cstab = 0.5
        else:
            cstab = 10
        eta2 = cstab*pow(deg,6.0)
        eta3 = cstab*pow(deg,4.0)
        if SM == True:
            eta4 = 0.0
        else:
            eta4 = cstab*pow(deg,4.0)

        x, y = SpatialCoordinate(mesh)

        # defining true solution, coefficient matrix A, and boundary condition function for benchmarks
        if prob == 0:
            rho = pow(x,2)+pow(y,2)
            u = pow(sin(pi*rho),2)
            c = cos(pi*rho)
            s = sin(pi*rho)
            trueu = u
            du0 = 4*pi*x*c*s
            du1 = 4*pi*y*c*s
            d2udxx = 4*pi*c*s+8*pi*pi*x*x*(c**2-s**2)
            d2udyy = 4*pi*c*s+8*pi*pi*y*y*(c**2-s**2)
            d2udxy = 8*pi*pi*x*y*(c**2-s**2)
            d4udxxxx = 24*pi**2*(c**2-s**2)-384*pi**3*x**2*c*s+128*pi**4*x**4*(s**2-c**2)
            d4udyyyy = 24*pi**2*(c**2-s**2)-384*pi**3*y**2*c*s+128*pi**4*y**4*(s**2-c**2)
            d4udxxyy = 8*pi**2*(c**2-s**2)-64*pi**3*rho*c*s+128*pi**4*x**2*y**2*(s**2-c**2)
            
            f = d4udxxxx+2.0*d4udxxyy+d4udyyyy
            g = u
            dg0 = du0
            dg1 = du1
            d2g00 = d2udxx
            d2g01 = d2udxy
            d2g11 = d2udyy
        
        elif prob == 1:
            py = pi*y
            px = pi*x
            u = pow(sin(px),2)*pow(sin(py),2)
            du0 = 2*pi*cos(px)*sin(px)*pow(sin(py),2)
            du1 = 2*pi*cos(py)*sin(py)*pow(sin(px),2)
            d2udxx = -2*pi*pi*pow(sin(px),2)*pow(sin(py),2)+2*pi*pi*pow(cos(px),2)*pow(sin(py),2)
            d2udyy = -2*pi*pi*pow(sin(px),2)*pow(sin(py),2)+2*pi*pi*pow(cos(py),2)*pow(sin(px),2)
            d2udxy = 4*pi*pi*sin(px)*cos(px)*sin(py)*cos(py)
            d4udxxxx = 8*pow(pi,4)*(pow(sin(px),2)-pow(cos(px),2))*pow(sin(py),2)
            d4udyyyy = 8*pow(pi,4)*(pow(sin(py),2)-pow(cos(py),2))*pow(sin(px),2)
            d4udxxyy = 4*pow(pi,4)*pow(sin(px),2)*pow(sin(py),2)-4*pow(pi,4)*pow(sin(px),2)*pow(cos(py),2)-4*pow(pi,4)*pow(sin(py),2)*pow(cos(px),2)+4*pow(pi,4)*pow(cos(px),2)*pow(cos(py),2)
            f = d4udxxxx+2.0*d4udxxyy+d4udyyyy
            g = u

        def a(u,v):
            if SM == True:
                a = (u.dx(0).dx(0)+u.dx(1).dx(1))*(v.dx(0).dx(0)+v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
            else:
                a = (u.dx(0).dx(0)*v.dx(0).dx(0)+u.dx(0).dx(1)*v.dx(0).dx(1)+u.dx(1).dx(0)*v.dx(1).dx(0)+u.dx(1).dx(1)*v.dx(1).dx(1))*dx(mesh,degree = quad_deg)
            return a
        
        def B2in(u,v):
            B = ((avg(u.dx(0).dx(0).dx(0)+u.dx(1).dx(1).dx(0))*n[0]('+')+avg(u.dx(1).dx(1).dx(1)+u.dx(0).dx(0).dx(1))*n[1]('+'))*jump(v)\
                 -avg(u.dx(0).dx(0)+u.dx(1).dx(1))*inner(jump(grad(v)),n('+'))\
                 )\
                *dS(mesh,degree = quad_deg)
            return B
        def B2out(u,v):
            B = (((u.dx(0).dx(0).dx(0)+u.dx(1).dx(1).dx(0))*n[0]+(u.dx(0).dx(0).dx(1)+u.dx(1).dx(1).dx(1))*n[1])*(v)\
                 -(u.dx(0).dx(0)+u.dx(1).dx(1))*inner((grad(v)),n)\
                 )\
                *ds(mesh,degree = quad_deg)
            return B
        def B2(u,v):
            B = B2in(u,v)+B2out(u,v)
            return B
                         
        def C1(u,v):
            C = (avg(u).dx(0).dx(0)+avg(u).dx(1).dx(1)\
                -avg(u).dx(0).dx(0)*n[0]('+')*n[0]('+')\
                -avg(u).dx(1).dx(0)*n[1]('+')*n[0]('+')\
                -avg(u).dx(0).dx(1)*n[0]('+')*n[1]('+')\
                -avg(u).dx(1).dx(1)*n[1]('+')*n[1]('+')\
                )*\
                (inner(jump(grad(v)),n('+')))\
                *dS(mesh,degree = quad_deg)
            return C
        def C2(u,v):
            C = (n[0]('+').dx(0)+n[1]('+').dx(1)-n[0]('+')*(n[0]('+').dx(0)*n[0]('+')+n[0]('+').dx(1)*n[1]('+'))-n[1]('+')*(n[1]('+').dx(0)*n[0]('+')+n[1]('+').dx(1)*n[1]('+')))*(inner(avg(grad(u)),n('+')))*(inner(jump(grad(v)),n('+')))*dS(mesh,degree = quad_deg)
            return C
        def C3(u,v):
            C = -(\
                 (((n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(0) \
                -n[0]('+')*n[0]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(0) \
                -n[0]('+')*n[1]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(1) \
                )) \
                * \
                ( \
                v.dx(0)('+')-v.dx(0)('-')-n[0]('+')*n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))\
                -n[0]('+')*n[1]('+')*(v.dx(1)('+')-v.dx(1)('-'))\
                )\
                +(\
                ((n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(1) \
                -n[1]('+')*n[0]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(0) \
                -n[1]('+')*n[1]('+')*(n[0]('+')*avg(u.dx(0))+n[1]('+')*avg(u.dx(1))).dx(1) \
                )) \
                * \
                ( \
                v.dx(1)('+')-v.dx(1)('-')-n[1]('+')*n[0]('+')*(v.dx(0)('+')-v.dx(0)('-'))\
                -n[1]('+')*n[1]('+')*(v.dx(1)('+')-v.dx(1)('-'))\
                )\
                )\
                *dS(mesh,degree = quad_deg)
            return C
        def Q(u,v):
            Q1 = (u[0]*n[0]('+').dx(0)*v[0]+u[0]*n[1]('+').dx(0)*v[1]+u[1]*n[0]('+').dx(1)*v[0]+u[1]*n[1]('+').dx(1)*v[1])
            return Q1
        #def C4(u,v):
        #C = bdry_indicator*Q(grad(avg(u))-n('+')*inner(grad(avg(u)),n('+')),jump(grad(v))-n('+')*inner(jump(grad(v)),n('+')))*dS(mesh,degree = quad_deg)
        #return C
        #def C4(u,v):
        #C = bdry_indicator*Q(n('+')*inner(avg(grad(u)),n('+')),jump(grad(v))-n('+')*inner(jump(grad(v)),n('+')))*dS(mesh,degree = quad_deg)
        #return C
        def C4(u,v):
            C = Q(avg(grad(u)),jump(grad(v))-n('+')*inner(jump(grad(v)),n('+')))*dS(mesh,degree = quad_deg)
            return C
        def C(u,v):
            CC = C1(u,v)+C2(u,v)+C3(u,v)+C4(u,v)
            return CC
        # Jump penalty operator
        def J1in(u,v):
            Jin = (eta2/hin**3)*jump(u)*jump(v)*dS(mesh,degree = quad_deg)
            return Jin
        def J1out(u,v):
            Jout = (eta2/hout**3)*u*v*ds(mesh,degree = quad_deg)
            return Jout
        def J2in(u,v):
            Jin = (eta3/hin)*inner(jump(grad(u)),n('+'))*inner(jump(grad(v)),n('+'))*dS(mesh,degree = quad_deg)
            return Jin
        def J2out(u,v):
            Jout = (eta3/hout)*inner(grad(u),n)*inner(grad(v),n)*ds(mesh,degree = quad_deg)
            return Jout
        def J3in(u,v):
            Jin = (eta4/hin)*inner(jump(grad(u))-inner(jump(grad(u)),n('+'))*n('+'),jump(grad(v))-inner(jump(grad(v)),n('+'))*n('+'))*dS(mesh,degree = quad_deg)
            return Jin
        def J3out(u,v):
            Jout = (eta4/hout)*inner(grad(u)-inner(grad(u),n)*n,grad(v)-inner(grad(v),n)*n)*ds(mesh,degree = quad_deg)
            return Jout
        def J(u,v):
            J1 = J1in(u,v)+J1out(u,v)+J2in(u,v)+J2out(u,v)+J3in(u,v)+J3out(u,v)
            return J1
        
        # defining bilinear form
        A = a(U,v)\
            +B2(U,v)\
            +B2(v,U)\
            +J(U,v)
        if SM == False:
            A += C(U,v)+C(v,U)
        # defining linear form
        L = (f)*(v)*dx(mesh,degree = quad_deg)

        # altering the linear form for the inhomogeneous BC case
        if inhomogeneous == True:
            L += B2out(v,g) + J1out(g,v) + J2out(g,v)
        # defining solution function
        U = Function(FES)

        # begin timing of linear system solve
        t = time()

        # solving linear system
        solve(A == L, U,
              solver_parameters = {
              "snes_type": "newtonls",
              "ksp_type": "preonly",
              "pc_type": "lu",
              "snes_monitor": False,
              "snes_rtol": 1e-16,
              "snes_atol": 1e-25})

        # end timing of linear system solve
        tt.append(time()-t)
        hm.append(h)
        gradu = as_vector([du0,du1])
        nerr = as_vector([n[0],n[1]])
        # calculating errors
        errorL2 = (U-u)**2*dx(mesh,degree = quad_deg)
        errorH1 = ((U.dx(0)-du0)**2+(U.dx(1)-du1)**2)*dx(mesh,degree = quad_deg)
        errorH2 = ((U.dx(0).dx(0)-d2udxx)**2+(U.dx(1).dx(1)-d2udyy)**2+(U.dx(0).dx(1)-d2udxy)**2+(U.dx(1).dx(0)-d2udxy)**2)*dx(mesh,degree = quad_deg)+J(U-u,U-u)
        errorEN = (U.dx(0).dx(0)+U.dx(1).dx(1)-d2udxx-d2udyy)**2*dx(mesh,degree = quad_deg)
        
        eL2 = sqrt(assemble(errorL2))
        eH1 = sqrt(assemble(errorH1))
        eH2 = sqrt(assemble(errorH2))
        eEN = sqrt(assemble(errorEN))
        e_L2.append(eL2)
        e_H1.append(eH1)
        e_H2.append(eH2)
        e_EN.append(eEN)

        # obtain number of DoFs
        ndof.append(U.vector().array().size)

    EOCL2 = []
    EOCH1 = []
    EOCH2 = []
    EOCEN = []
    EOClinf = []
    EOCl2proj = []


    EOCL2.append(0)
    EOCH1.append(0)
    EOCH2.append(0)
    EOCEN.append(0)
    EOClinf.append(0)
    EOCl2proj.append(0)

    #Calcuating error orders of convergence.
    for k in range(1,len(e_L2)):
        EOCL2.append(ln(e_L2[k-1]/(e_L2[k]))/ln(hm[k-1]/hm[k]))
        EOCH1.append(ln(e_H1[k-1]/(e_H1[k]))/ln(hm[k-1]/hm[k]))
        EOCH2.append(ln(e_H2[k-1]/(e_H2[k]))/ln(hm[k-1]/hm[k]))
        EOCEN.append(ln(e_EN[k-1]/(e_EN[k]))/ln(hm[k-1]/hm[k]))
    k = 0

    # outputting jump stabilisation constants
    print("eta2 = ",eta2)
    print("eta3 = ",eta3)
    print("eta4 = ",eta4)

    # outputting error results
    for k in range(len(e_L2)):
        print( "NDoFs = ", ndof[k])
        print( "Mesh size = ", hm[k])
        print( "run time = ",tt[k])
        print( "||u - u_h||_0 = ", e_L2[k], "   EOC = ", EOCL2[k])
        print( "||u - u_h||_1 = ", e_H1[k], "   EOC = ", EOCH1[k])
        print( "||u - u_h||_2 = ", e_H2[k], "   EOC = ", EOCH2[k])
        print( "||u - u_h||_EN = ", e_EN[k], "   EOC = ", EOCEN[k])
        k = k+1
    # returning errors, NDofs, runtimes, EOCs and polynomial degree for the creation of data files
    return e_L2, e_H1, e_H2, e_EN, ndof, hm, tt,  EOCL2, EOCH1, EOCH2, EOCEN, deg
# solving problem for degrees 2, 3 and 4, and saving data to text file
for degg in [2,3,4]:
    e_L2, e_H1, e_H2, e_EN, ndof, hm, tt,  EOCL2, EOCH1, EOCH2, EOCEN, deg = model(10,degg)
    print( "polynomial degree: ", deg)
    print( "integral degree: ", quad_deg)
    print( "experiment no. : ", prob)
    out_name1 = "Curved-biharmonic/e_L2p%i.txt" %deg
    out_name2 = "Curved-biharmonic/e_H1p%i.txt" %deg
    out_name3 = "Curved-biharmonic/e_H2p%i.txt" %deg
    out_name4 = "Curved-biharmonic/e_ENp%i.txt" %deg
    out_name5 = "Curved-biharmonic/EOCL2p%i.txt" %deg
    out_name6 = "Curved-biharmonic/EOCH1p%i.txt" %deg
    out_name7 = "Curved-biharmonic/EOCH2p%i.txt" %deg
    out_name8 = "Curved-biharmonic/EOCENp%i.txt" %deg
    out_name9 = "Curved-biharmonic/dofsp%i.txt" %deg
    out_name10 = "Curved-biharmonic/meshsize.txt"
    out_name11 = "Curved-biharmonic/timesp%i.txt" %deg


    np.savetxt(out_name1,e_L2,fmt = '%s')
    np.savetxt(out_name2,e_H1,fmt = '%s')
    np.savetxt(out_name3,e_H2,fmt = '%s')
    np.savetxt(out_name4,e_EN,fmt = '%s')
    np.savetxt(out_name5,EOCL2,fmt = '%s')
    np.savetxt(out_name6,EOCH1,fmt = '%s')
    np.savetxt(out_name7,EOCH2,fmt = '%s')
    np.savetxt(out_name8,EOCEN,fmt = '%s')
    np.savetxt(out_name9,ndof,fmt = '%s')
    np.savetxt(out_name10,hm,fmt = '%s')
    np.savetxt(out_name11,tt,fmt = '%s')
