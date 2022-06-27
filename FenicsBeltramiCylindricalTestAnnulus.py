#############################################################################################
#
#               This simple code computes Beltrami fields in an annulus for an axisymmetric 
#		toroidal geometry with circular cross section with FEniCS
#               The numerical solution is compared, in the large aspect ratio limit to an exact 
#		solution by M.J. Hole et al., Nuclear Fusion 47, 746 (2007)
#
#               % Copyright (C) 2022: Antoine Cerfon
#               Contact: cerfon@cims.nyu.edu
# 
#               This program is free software; you can redistribute it and/or modify 
#               it under the terms of the GNU General Public License as published by 
#               the Free Software Foundation; either version 2 of the License, or 
#               (at your option) any later version.  This program is distributed in 
#               the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
#               even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
#               PARTICULAR PURPOSE.  See the GNU General Public License for more 
#               details. You should have received a copy of the GNU General Public 
#               License along with this program; if not, see <http://www.gnu.org/licenses/>.
#
############################################################################################


from fenics import *
from mshr import Polygon, generate_mesh
import matplotlib.pyplot as plt
import mshr
import numpy as np
import scipy.special as sp
from ufl import bessel_J
from ufl import bessel_Y

# Define values of major and minor axes
R0=float(100.0)
amin1 = float(1.0)
amin2 = float(0.6)
eps = float(amin1/R0)#Inverse aspect ratio

# Define value of Beltrami constant
mu = float(1.0)

# Define value of poloidal flux
psipol1 = float(100.0)
psipol2 = float(100.0)

# Define value of toroidal flux
psitor = float(20.0)

# Create mesh for Circular domain
domain = mshr.Circle(Point(R0,0.),amin1) - mshr.Circle(Point(R0,0.),amin2)
mesh = mshr.generate_mesh(domain, 100, "cgal")

# Normalize toroidal flux for Lagrange multiplier approach
surface_area = assemble(Constant(1.0)*dx(mesh))
psi_norm = float(psitor/surface_area)

# Define function space

Vtest = FunctionSpace(mesh, 'P', 1)
V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
mixed_element = MixedElement([V, R])
X = FunctionSpace(mesh, mixed_element)

# Define value of poloidal flux on inner and outer surface
u_Out = psipol1+psipol2
u_In = psipol1

# Define inner surface
def inner_boundary(x, on_boundary):
    tol = 1E-2   # tolerance for coordinate comparisons
    return on_boundary and near(sqrt((x[0]-R0)**2 + x[1]**2), amin2,tol)
Gamma_In = DirichletBC(X.sub(0), u_In, inner_boundary)

# Define outer surface
def outer_boundary(x, on_boundary):
    tol = 1E-2   # tolerance for coordinate comparisons
    return on_boundary and near(sqrt((x[0]-R0)**2 + x[1]**2), amin1,tol)
Gamma_Out = DirichletBC(X.sub(0), u_Out, outer_boundary)

# Dirichlet boundary conditions on each surface
bcs = [Gamma_Out, Gamma_In]

x = Expression('x[0]',degree = 1)
y = Expression('x[1]',degree = 1)

# Define variational problem for generic homogeneous solution
(u, k) = TrialFunctions(X)
(v, r) = TestFunctions(X)
a = (-dot(grad(u), grad(v))+mu**2*u*v)/x*dx+mu*k*v/x*dx+(mu*u+k)/x*r*dx
L = Constant(0.0)*v*dx+Constant(psi_norm)*r*dx

# Compute solution
xfun = Function(X)
solve(a==L, xfun, bcs)

psi, ksol = xfun.split()

# Construct components of the magnetic field from the poloidal flux
W = VectorFunctionSpace(mesh, 'Lagrange', 1)

# Construct components of the magnetic field
grad_psi = project(grad(psi), W)
dpsidr, dpsidz = grad_psi.split(deepcopy=True) # extract components of the gradient
Br = project(-1/x*dpsidz,Vtest)
Bz = project(1/x*dpsidr,Vtest)
Bphi = project((mu*psi+ksol)/x,Vtest) 

# Construct exact straight cylinder solution for comparison

list1=[float(bessel_J(0,abs(mu)*amin2)), float(bessel_Y(0,abs(mu)*amin2)), -1/mu]
list2=[float(bessel_J(0,abs(mu)*amin1)),float(bessel_Y(0,abs(mu)*amin1)),-1/mu]
list3=[float(amin1*bessel_J(1,abs(mu)*amin1)-amin2*bessel_J(1,abs(mu)*amin2)),float(amin1*bessel_Y(1,abs(mu)*amin1)-amin2*bessel_Y(1,abs(mu)*amin2)),0]

LeftMat = np.matrix([list1,list2,list3])
RightVec = np.array([psipol1,psipol1+psipol2,R0*psitor*mu/(2*pi*abs(mu))])
# Note: Need to multiply toroidal flux by R0 because Bz,cylindrical = -1/R0 g(psi) (see Freidberg, Ideal MHD, Eq. (6.61) p.140)
TestVec = np.linalg.solve(LeftMat,RightVec)
# Flux function
psi_cylind = TestVec[0]*bessel_J(0,abs(mu)*sqrt((x-R0)**2+y**2))+TestVec[1]*bessel_Y(0,abs(mu)*sqrt((x-R0)**2+y**2))-TestVec[2]/mu
psicylind = project(psi_cylind, Vtest)
psinumproject = project(psi,Vtest)

# Components of the magnetic field
# Btheta
Btheta_exact=-abs(mu)/R0*(TestVec[0]*bessel_J(1,mu*sqrt((x-R0)**2+y**2))+TestVec[1]*bessel_Y(1,mu*sqrt((x-R0)**2+y**2)))# Freidberg Eq. (6.61): 1/R0 dpsi/dr
Bthetaexact = project(Btheta_exact, Vtest)
B_theta = (-y/sqrt((x-R0)**2+y**2)*Br+(x-R0)/sqrt((x-R0)**2+y**2)*Bz)
Btheta = project(B_theta, Vtest)
Btheta_diff=Btheta-Bthetaexact
Bthetadiff = project(Btheta_diff,Vtest)

#B_rho
B_rho = (x-R0)/sqrt((x-R0)**2+y**2)*Br+y/sqrt((x-R0)**2+y**2)*Bz
Brho = project(B_rho, Vtest)

# Bz
BzCyl_exact = (mu*psi_cylind+TestVec[2])/R0# Freidberg Eq. (6.61): Bz=-g(psi)/R0 => Define Bz to compare as Bz_Cyl=g(psi)/R0 so field is pointing in the same direction
BzCylexact = project(BzCyl_exact, Vtest)
BzCyl_diff = Bphi-BzCylexact
BzCyldiff = project(BzCyl_diff,Vtest)

# Plot flux functions for toroidal solution and straight cylinder exact solution

plt.figure()
plt.subplot(1,3,1)
a1=plot(psinumproject,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Poloidal $\Psi$ from torus calculation, for $\epsilon$='+str(eps))
plt.colorbar(a1)
plt.subplot(1,3,2)
a2=plot(psicylind,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Exact poloidal $\Psi$ for straight circular cylinder')
plt.colorbar(a2)
plt.subplot(1,3,3)
a3=plot(psicylind-psinumproject,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Difference between poloidal $\Psi$ from numerical toroidal calculation \n and exact cylindrical calculation, for $\epsilon$='+str(eps))
plt.colorbar(a3)

plt.figure()
plt.subplot(1,3,1)
b1=plot(Br,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{R}$')
plt.colorbar(b1)
plt.subplot(1,3,2)
b2=plot(Bz,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{Z}$')
plt.colorbar(b2)
plt.subplot(1,3,3)
b3=plot(Bphi,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{\phi}$')
plt.colorbar(b3)

plt.figure()
plt.subplot(2,3,1)
c1=plot(Btheta,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{\\theta}$ from torus numerical calculation, for $\epsilon$='+str(eps))
plt.colorbar(c1)
plt.subplot(2,3,2)
c2=plot(Bthetaexact,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Exact $B_{\\theta}$ for straight circular cylinder, for $\epsilon$='+str(eps))
plt.colorbar(c2)
plt.subplot(2,3,3)
c3=plot(Bthetadiff,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Difference for $\epsilon$='+str(eps))
plt.colorbar(c3)
plt.subplot(2,3,4)
c4=plot(Bphi,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{\phi}$ from torus numerical calculation, for $\epsilon$='+str(eps))
plt.colorbar(c4)
plt.subplot(2,3,5)
c5=plot(BzCylexact,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Exact $B_{z}$ for straight circular cylinder, for $\epsilon$='+str(eps))
plt.colorbar(c5)
plt.subplot(2,3,6)
c6=plot(BzCyldiff,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Difference for $\epsilon$='+str(eps))
plt.colorbar(c6)

plt.figure()
e1=plot(Brho,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{\\rho}$ from torus numerical calculation, for $\epsilon$='+str(eps))
plt.colorbar(e1)
               
plt.show()    


# Save solution to file in VTK format
#vtkfile = File('beltrami/solution.pvd')
#vtkfile << u
