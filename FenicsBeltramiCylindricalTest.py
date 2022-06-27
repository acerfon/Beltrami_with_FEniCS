#############################################################################################
#
#               This simple code computes Beltrami fields in an axisymmetric 
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

# Define values of major and minor axes
R0=Constant(100.0)
amin = Constant(1.0)
eps = float(amin/R0)# Inverse aspect ratio

# Define value of poloidal flux
psipol = Constant(200.0)

# Create mesh for Circular domain
domain = mshr.Circle(Point(R0,0.),amin)
mesh = mshr.generate_mesh(domain, 100, "cgal")

# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Define value of Beltrami constant
mu = Constant(1.0)

# Define boundary condition - Needs to be set to 1 for construction of the generic homogeneous solution
u_D = Constant(1.0)

x = Expression('x[0]',degree = 1)
y = Expression('x[1]',degree = 1)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem for generic homogeneous solution
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
a = (-dot(grad(u), grad(v))+mu**2*u*v)/x*dx
L = f*v*dx

# Compute homogeneous solution
u = Function(V)
solve(a == L, u, bc)

# plt.figure()
# a1=plot(u,mode='color',cmap='YlOrRd')
# plt.xlabel("R")
# plt.ylabel("Z")
# plt.colorbar(a1)

# Find value of psi homogeneous at the magnetic axis
vertex_values_u = u.compute_vertex_values(mesh)
umax = np.max(vertex_values_u)

# Construct full flux function
psi = project(psipol/(1-umax)*(u-umax),V)
psi_cylind = psipol*(1-bessel_J(0,abs(mu)*sqrt((x-R0)**2+y**2)))/(1-bessel_J(0,abs(mu)*amin))
psicylind = project(psi_cylind, V)
psi_diff = psi-psicylind
psidiff = project(psi_diff,V)

# Plot flux function and Test solver for very high aspect ratio tokamak with circular cross section
plt.figure()
plt.subplot(1,3,1)
b1=plot(psi,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Poloidal $\Psi$ from torus calculation, for $\epsilon$='+str(eps))
plt.colorbar(b1)
plt.subplot(1,3,2)
b2=plot(psicylind,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.colorbar(b2)
plt.title('Exact poloidal $\Psi$ for straight circular cylinder')
plt.subplot(1,3,3)
b3=plot(psidiff,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.colorbar(b3)
plt.title('Difference between poloidal $\Psi$ from numerical toroidal calculation \n and exact cylindrical calculation, for $\epsilon$='+str(eps))


W = VectorFunctionSpace(mesh, 'P', 1)

# Construct components of the magnetic field
grad_psi = project(grad(psi), W)
dpsidr, dpsidz = grad_psi.split(deepcopy=True) # extract components of the gradient
Br = project(-1/x*dpsidz,V)
Bz = project(1/x*dpsidr,V)
Bphi = project(mu*psi/x+mu/x*psipol*umax/(1-umax),V)             

# Plot components of the magnetic field

plt.figure()
plt.subplot(1,3,1)
c1=plot(Br,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{R}$')
plt.colorbar(c1)
plt.subplot(1,3,2)
c2=plot(Bz,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{Z}$')
plt.colorbar(c2)
plt.subplot(1,3,3)
c3=plot(Bphi,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{\phi}$')
plt.colorbar(c3)

# Test solver for very high aspect ratio tokamak with circular cross section
Btheta_exact=(1/R0)*abs(mu)*psipol*bessel_J(1,mu*sqrt((x-R0)**2+y**2))/(1-bessel_J(0,abs(mu)*amin))# Freidberg Eq. (6.61): 1/R0 dpsi/dr
Bthetaexact = project(Btheta_exact, V)
B_theta = (-y/sqrt((x-R0)**2+y**2)*Br+(x-R0)/sqrt((x-R0)**2+y**2)*Bz)
Btheta = project(B_theta, V)
Btheta_diff=Btheta-Bthetaexact
Bthetadiff = project(Btheta_diff,V)

BzCyl_exact = -(1/R0)*mu*psipol*bessel_J(0,mu*sqrt((x-R0)**2+y**2))/(1-bessel_J(0,abs(mu)*amin))# Freidberg Eq. (6.61): Bz=-g(psi)/R0 => Define Bz to compare as Bz_Cyl=g(psi)/R0 so field is pointing in the same direction
BzCylexact = project(BzCyl_exact, V)
BzCyl_diff = Bphi-BzCylexact
BzCyldiff = project(BzCyl_diff,V)

B_rho = (x-R0)/sqrt((x-R0)**2+y**2)*Br+y/sqrt((x-R0)**2+y**2)*Bz
Brho = project(B_rho, V)

plt.figure()
plt.subplot(2,3,1)
d1=plot(Btheta,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{\\theta}$ from torus numerical calculation, for $\epsilon$='+str(eps))
plt.colorbar(d1)
plt.subplot(2,3,2)
d2=plot(Bthetaexact,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Exact $B_{\\theta}$ for straight circular cylinder, for $\epsilon$='+str(eps))
plt.colorbar(d2)
plt.subplot(2,3,3)
d3=plot(Bthetadiff,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Difference for $\epsilon$='+str(eps))
plt.colorbar(d3)
plt.subplot(2,3,4)
e2=plot(Bphi,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{\phi}$ from torus numerical calculation, for $\epsilon$='+str(eps))
plt.colorbar(e2)
plt.subplot(2,3,5)
e1=plot(BzCylexact,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Exact $B_{z}$ for straight circular cylinder, for $\epsilon$='+str(eps))
plt.colorbar(e1)
plt.subplot(2,3,6)
e3=plot(BzCyldiff,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('Difference for $\epsilon$='+str(eps))
plt.colorbar(e3)

plt.figure()
f1=plot(Brho,mode='color',cmap='YlOrRd')
plt.xlabel("R")
plt.ylabel("Z")
plt.title('$B_{\\rho}$ from torus numerical calculation, for $\epsilon$='+str(eps))
plt.colorbar(f1)
                 
plt.show()    

# Save solution to file in VTK format
#vtkfile = File('beltrami/solution.pvd')
#vtkfile << u
