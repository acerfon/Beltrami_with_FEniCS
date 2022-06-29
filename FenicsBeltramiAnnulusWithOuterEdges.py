#############################################################################################
#
#               This simple code computes Beltrami fields in an axisymmetric 
#		toroidal annular geometry with FEniCS, with a pentagon as the outer cross section
#               and a circle as the inner cross section
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

# Define value of major and minor radii
R0=float(5.0)
amin1 = float(1.0)# effective minor radius of pentagon
amin2 = float(0.5)# minor radius of inner disc
eps = float(amin1/R0)# Inverse aspect ratio

# Define value of Beltrami constant
mu = float(1.0)

# Define value of poloidal flux
psipol1 = float(10.0) #poloidal flux on inner boundary
psipol2 = float(10.0) #the poloidal flux on the outer boundary is psipol1+psipol2

# Define value of toroidal flux
psitor = float(4.0)

# Create mesh for annular domain
domain = Polygon([Point(amin1+R0,0),Point(R0+amin1*cos(2*pi/5),amin1*sin(2*pi/5)),Point(R0+amin1*cos(4*pi/5),amin1*sin(4*pi/5)),Point(R0+amin1*cos(6*pi/5),amin1*sin(6*pi/5)),Point(R0+amin1*cos(8*pi/5),amin1*sin(8*pi/5))]) - mshr.Circle(Point(R0,0.),amin2)
mesh = mshr.generate_mesh(domain, 100, "cgal")

# Normalize toroidal flux for Lagrange multiplier approach
surface_area = assemble(Constant(1.0)*dx(mesh))
psi_norm = float(psitor/surface_area)

# Define function spaces

Vtest = FunctionSpace(mesh, 'P', 1)
V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
mixed_element = MixedElement([V, R])
X = FunctionSpace(mesh, mixed_element)

# Define value of poloidal flux on inner and outer surface
u_Out = psipol1+psipol2
u_In = psipol1

# First impose outer boundary condition on all boundaries of domain (easier, since it would be cumbersome to define edges of pentagon individually)
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(X.sub(0), u_Out, boundary)

# Now fix the boundary condition for the inner boundary
# Define inner surface
def inner_boundary(x, on_boundary):
    tol = 1E-4   # tolerance for coordinate comparisons
    return on_boundary and near(sqrt((x[0]-R0)**2 + x[1]**2), amin2,tol)
Gamma_In = DirichletBC(X.sub(0), u_In, inner_boundary)

# Dirichlet boundary conditions on each surface
bcs = [bc,Gamma_In]

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
psinumproject = project(psi,Vtest)

# Construct components of the magnetic field from the poloidal flux
W = VectorFunctionSpace(mesh, 'Lagrange', 1)
# Construct components of the magnetic field
grad_psi = project(grad(psi), W)
dpsidr, dpsidz = grad_psi.split(deepcopy=True) # extract components of the gradient
Br = project(-1/x*dpsidz,Vtest)
Bz = project(1/x*dpsidr,Vtest)
Bphi = project((mu*psi+ksol)/x,Vtest) 

# Plot flux function and components of B

plt.figure()
a1=plot(psinumproject,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Poloidal $\Psi$ from torus calculation, for $\epsilon$='+str(eps))
plt.colorbar(a1)

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
                 
plt.show()    


# Save solution to file in VTK format
#vtkfile = File('beltrami/solution.pvd')
#vtkfile << u
