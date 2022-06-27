#############################################################################################
#
#               This simple code computes Beltrami fields in an axisymmetric 
#		toroidal geometry with pentagonal cross section with FEniCS
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

# Define values of major and minor radius
R0=Constant(5.0)
amin = Constant(1.0)
eps = float(amin/R0)# Inverse aspect ratio

# Define value of poloidal flux
psipol = Constant(50.0)

# Pentagonal domain
domain = Polygon([Point(amin+R0,0),Point(R0+amin*cos(2*pi/5),amin*sin(2*pi/5)),Point(R0+amin*cos(4*pi/5),amin*sin(4*pi/5)),Point(R0+amin*cos(6*pi/5),amin*sin(6*pi/5)),Point(R0+amin*cos(8*pi/5),amin*sin(8*pi/5))])
mesh = generate_mesh(domain,80)

# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Define value of Beltrami constant
mu = Constant(2.0)

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

# Find value of psi homogeneous at the magnetic axis
vertex_values_u = u.compute_vertex_values(mesh)
umax = np.max(vertex_values_u)

# Construct full flux function
psi = project(psipol/(1-umax)*(u-umax),V)

# Plot flux function
b1=plot(psi,mode='color',cmap='YlOrRd')
plt.title('Poloidal $\Psi$ for $\epsilon$='+str(eps))
plt.xlabel("R")
plt.ylabel("Z")
plt.colorbar(b1)

W = VectorFunctionSpace(mesh, 'P', 1)
# Construct components of the magnetic field
grad_psi = project(grad(psi), W)
dpsidr, dpsidz = grad_psi.split(deepcopy=True) # extract components of the gradient
Br = project(-1/x*dpsidz,V)
Bz = project(1/x*dpsidr,V)
Bphi = project(mu*psi/x+mu/x*psipol*umax/(1-umax),V)             

## Plot components of the magnetic field

plt.figure()
plt.subplot(1,3,1)
c1=plot(Br,mode='color',cmap='YlOrRd')
plt.title('$B_{R}$')
plt.xlabel("R")
plt.ylabel("Z")
plt.colorbar(c1)
plt.subplot(1,3,2)
c2=plot(Bz,mode='color',cmap='YlOrRd')
plt.title('$B_{Z}$')
plt.xlabel("R")
plt.ylabel("Z")
plt.colorbar(c2)
plt.subplot(1,3,3)
c3=plot(Bphi,mode='color',cmap='YlOrRd')
plt.title('$B_{\phi}$')
plt.xlabel("R")
plt.ylabel("Z")
plt.colorbar(c3)
                 
plt.show()    

# Save solution to file in VTK format
#vtkfile = File('beltrami/solution.pvd')
#vtkfile << u
