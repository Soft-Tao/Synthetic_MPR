"""
***Synthetic MPR***
    > n-p transition efficiency are not considered in this version...
Author: Xutao Xu
Date: June 18th, 2025
"""

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import physical_constants

class Aperture:
    def __init__(self, r, theta, phi, l1, l2, d):
        # Coordinates of Aperture's center
        self.r = r # [m]
        self.theta = theta*np.pi/180 # [rad], must be 90
        self.phi = phi*np.pi/180 # [rad]
        # Size params of Aperture (rectangle)
        self.l1 = l1 # [m]
        self.l2 = l2 # [m]
        self.d = d # [m]

class Particle:
    # starting_E in [MeV], (E, theta [degree], phi [degree]) to (vx, vy, vz) [m/s]
    def __init__(self, starting_posX, starting_posY, starting_posZ, starting_E, starting_theta, starting_phi, particle_name, neutron_E):
        self.pos = np.array([starting_posX, starting_posY, starting_posZ])
        self.tof = 0
        self.neutron_E = neutron_E # the energy of the incident neutron which gave birth to this particle

        self.IsArrived = False # if True, the particle has striked the Focal Plane

        if particle_name == "proton":
            self.m = physical_constants["proton mass"][0] # [kg]
            self.q = physical_constants["elementary charge"][0] # [C]
        elif particle_name == "alpha":
            self.m = 4*physical_constants["proton mass"][0] # [kg]
            self.q = 2*physical_constants["elementary charge"][0] # [C]
        else:
            raise ValueError("Invalid particle name")
        
        # (E [MeV], theta [degree], phi [degree]) to (vx, vy, vz) [m/s]
        the_ = starting_theta*np.pi/180
        phi_ = starting_phi*np.pi/180
        vel_ = np.sqrt(2*starting_E*1e6*physical_constants["electron volt"][0]/self.m)
        self.vel = np.array([vel_*np.sin(the_)*np.cos(phi_), vel_*np.sin(the_)*np.sin(phi_), vel_*np.cos(the_)])

        self.name = particle_name
        if IsTraceSaved:
            self.trace = [[self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.tof]]
        else:
            self.trace = None
        
    def Boris_push(self, BfieldVec, dt):
        if not self.IsArrived:
            A = self.q*dt/(2*self.m)
            # Only magnetic field is considered, to form the coefficient matrix:
            M = np.array([[1, -A*BfieldVec[2], A*BfieldVec[1]], [A*BfieldVec[2], 1, -A*BfieldVec[0]], [-A*BfieldVec[1], A*BfieldVec[0], 1]])

            # v_{n+1} = M^{-1} \cdot v_{n}
            self.vel = np.matmul(np.linalg.inv(M), self.vel)
            # r_{n+1} = r_{n} + v_{n+1} \cdot dt
            old_pos = self.pos 
            self.pos = self.pos + self.vel*dt
            self.tof += dt

            intersection_pos = line_segment_intersection(old_pos, self.pos, B1=[FocalPlane1.x1, FocalPlane1.y1], B2=[FocalPlane1.x2, FocalPlane1.y2])
            if intersection_pos is not None:
                self.IsArrived = True
                
                # Record the intersection point
                l = np.sqrt((intersection_pos[0]-FocalPlane1.x2)**2 + (intersection_pos[1]-FocalPlane1.y2)**2)
                FocalPlane1.record_strike(l=l, z=intersection_pos[2], neutron_E=self.neutron_E)
            else:
                pass
        else:
            pass

    def record_pos(self):
        if IsTraceSaved:
            self.trace.append([self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.tof])
        else:
            raise ValueError("You cannot record position if IsTraceSaved is False!")


# The geometric center of the NPtarget is always at (0, 0, 0)
# only rotates around Z axis
class NPtarget:
    def __init__(self, l1, l2, d, angle, target_name):
        self.l1 = l1 # [m]
        self.l2 = l2 # [m]
        self.d = d # [m]
        self.angle = angle*np.pi/180 # [rad], angle between target and x+ axis, [0, pi]

        self.ESP = None # Energy Stopping Power

        if target_name == "CH2":
            self.ESP = np.loadtxt(data_dir + "/ESP.dat") # ESP: [MeV/(g/cm^2)] - E: [MeV]
            self.density = 0.1 # [g/cm^3]
        else:
            raise ValueError("Unknown target name: " + target_name)

        self.ESP_interp = None

    def _initialize_interpolators(self):
        self.ESP_interp = RegularGridInterpolator((self.ESP[:,0],), self.ESP[:,1])
    
    def get_ESP_at(self, E):
        if self.ESP_interp is None:
            self._initialize_interpolators()
        return self.ESP_interp([[E]])[0]*self.density*100 # dE/dx: [Mev/m]
    
    def get_Energy_loss(self, starting_pos, starting_E, theta, phi):
        x,y,z = starting_pos 
        # rotate <self.angle>, clockwise
        x1 = x*np.cos(self.angle) + y*np.sin(self.angle)
        y1 = -x*np.sin(self.angle) + y*np.cos(self.angle)
        z1 = z

        # unit vector
        dx = np.sin(theta)*np.cos(phi)
        dy = np.sin(theta)*np.sin(phi)  
        dz = np.cos(theta)
        # unit vector in local coordinate system
        dx1 = dx*np.cos(self.angle) + dy*np.sin(self.angle)
        dy1 = -dx*np.sin(self.angle) + dy*np.cos(self.angle)
        dz1 = dz

        x_min, x_max, y_min, y_max, z_min, z_max = -self.l1/2, self.l1/2, -self.d/2, self.d/2, -self.l2/2, self.l2/2
        t_values = []
        if dx1 != 0:
            if dx1 > 0:
                t_x = (x_max - x1) / dx1
            else:
                t_x = (x_min - x1) / dx1
            t_values.append(t_x)
        if dy1 != 0:
            if dy1 > 0:
                t_y = (y_max - y1) / dy1
            else:
                t_y = (y_min - y1) / dy1
            t_values.append(t_y)
        if dz1 != 0:
            if dz1 > 0:
                t_z = (z_max - z1) / dz1
            else:
                t_z = (z_min - z1) / dz1
            t_values.append(t_z)
        t_min = min(t for t in t_values if t > 0) # length in target [m]
        print(f"travel length from birth to exit: {t_min:.6f} m")

        # calculate the energy loss
        # step by step, calculate energy loss at each step (more accurate)
        s0 = 0
        ds = self.d/1000 # [m], step length
        while (s0 < t_min):
            s0 += ds
            starting_E -= ds * self.get_ESP_at(starting_E)
            if starting_E < 0.1: # [MeV]
                starting_E = 0.1
                break
        E_exit = starting_E

        exit_x1 = x1 + t_min * dx1
        exit_y1 = y1 + t_min * dy1
        exit_z1 = z1 + t_min * dz1
        # find the exit point, rotate back
        exit_x = exit_x1*np.cos(self.angle) - exit_y1*np.sin(self.angle)
        exit_y = exit_x1*np.sin(self.angle) + exit_y1*np.cos(self.angle)
        exit_z = exit_z1

        return E_exit, [exit_x, exit_y, exit_z]
    
class FocalPlane:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.record = [] # strike points of particles, [[l, z, neutron_E], ...]
    def record_strike(self, l, z, neutron_E):
        self.record.append([l, z, neutron_E])
    def write_record(self, filename, test_particles: list):
        with open(filename, 'w') as f:
            f.write(f"{len(test_particles)} particles was generated, {len(self.record)} of them hit the focal plane.\n")
            for record in self.record:
                f.write(f"{record[0]:.6f} {record[1]:.6f} {record[2]:.2f}\n")
        f.close()

class MagneticField3D:
    def __init__(self, Xshift, Yshift, Zshift): 
        self.xgrid = None
        self.ygrid = None
        self.zgrid = None
        self.Nx = None
        self.Ny = None
        self.Nz = None
        self.Bx = None
        self.By = None
        self.Bz = None
        self.Xshift = Xshift
        self.Yshift = Yshift
        self.Zshift = Zshift
        self._Bx_interp = None
        self._By_interp = None
        self._Bz_interp = None

    def _initialize_interpolators(self):
        self._Bx_interp = RegularGridInterpolator((self.xgrid, self.ygrid, self.zgrid), self.Bx)
        self._By_interp = RegularGridInterpolator((self.xgrid, self.ygrid, self.zgrid), self.By)
        self._Bz_interp = RegularGridInterpolator((self.xgrid, self.ygrid, self.zgrid), self.Bz)

    def get_Bvector_at(self, x, y, z):
        # if the particle is outside the grid, return [0, 0, 0]
        if x < self.xgrid[0] or x > self.xgrid[-1] or y < self.ygrid[0] or y > self.ygrid[-1] or z < self.zgrid[0] or z > self.zgrid[-1]:
            return 0, 0, 0
        
        if self._Bx_interp is None:
            self._initialize_interpolators()
        Bx = self._Bx_interp([[x, y, z]])[0]
        By = self._By_interp([[x, y, z]])[0]
        Bz = self._Bz_interp([[x, y, z]])[0]
        return Bx, By, Bz
    
    def Read_Bfield1(self):
        # Cooridinates of MagFieldGrid's origin in the new coordinate system (Xshift, Yshift, Zshift) [m]
        self.xgrid = sio.loadmat(data_dir + "/BposX.mat")["B_posX"][:,0]/1000 + self.Xshift # [m]
        self.ygrid = sio.loadmat(data_dir + "/BposY.mat")["B_posY"][:,0]/1000 + self.Yshift # [m]
        self.zgrid = sio.loadmat(data_dir + "/BposZ.mat")["B_posZ"][:,0]/1000 + self.Zshift # [m]

        self.Nx = len(self.xgrid)
        self.Ny = len(self.ygrid)
        self.Nz = len(self.zgrid)
        print("Spatial Grid of MagField: Nx, Ny, Nz = ", self.Nx, self.Ny, self.Nz)

        self.Bx = sio.loadmat(data_dir + "/Bxfield.mat")["Bf_X"].reshape((self.Nx, self.Ny, self.Nz)) # [T]
        self.By = sio.loadmat(data_dir + "/Byfield.mat")["Bf_Y"].reshape((self.Nx, self.Ny, self.Nz)) # [T]
        self.Bz = sio.loadmat(data_dir + "/Bzfield.mat")["Bf_Z"].reshape((self.Nx, self.Ny, self.Nz)) # [T]
        print("MagField is loaded!")
    
    def Read_Bfield2(self, factor = 1):
        # Read MagField from .dat file: X, Y, Z, Bx, By, Bz [m, m, m, T, T, T] \n
        # Cooridinates of MagFieldGrid's origin in the new coordinate system (Xshift, Yshift, Zshift) [m]
        # Nx = 91, Ny = 205, Nz = 41
        data = np.loadtxt(data_dir + "/combined_field_upgrade.txt", skiprows=2)
        self.xgrid = data[:,0].reshape((41, 205, 91))[0,0,:]/1000 + self.Xshift # [m]
        self.ygrid = data[:,1].reshape((41, 205, 91))[0,:,0]/1000 + self.Yshift # [m]
        self.zgrid = data[:,2].reshape((41, 205, 91))[:,0,0]/1000 + self.Zshift # [m]

        self.Nx = 91
        self.Ny = 205
        self.Nz = 41
        print("Spatial Grid of MagField: Nx, Ny, Nz = ", self.Nx, self.Ny, self.Nz)

        self.Bx = np.transpose(data[:,3].reshape((41, 205, 91)), (2, 1, 0))*factor # [T]
        self.By = np.transpose(data[:,4].reshape((41, 205, 91)), (2, 1, 0))*factor # [T]
        self.Bz = np.transpose(data[:,5].reshape((41, 205, 91)), (2, 1, 0))*factor # [T] # can be manually changed by setting factor != 1
        print("MagField is loaded!")

    # This fuction is mainly used for testing
    # Loading arbitrary Bfield with xgrid, ygrid, zgrid [Narray of size Nx, Ny and Nz] and Bx, By, Bz [Narray of size (Nx, Ny, Nz)]
    def Arbitrary_Bfield(self, xgrid, ygrid, zgrid, Bx, By, Bz):
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.zgrid = zgrid
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        print("MagField is loaded!")

def Beam_init(Nparticle, target: NPtarget, Neutron_E, Neutron_the, Neutron_phi, IsParallel = False):

    Particle_lst = []
    for i in range(Nparticle):

        # Get POSITION, it could be any point in target
        x1 = np.random.uniform(-target.l1/2, target.l1/2)
        y1 = np.random.uniform(-target.d/2, target.d/2)
        z = np.random.uniform(-target.l2/2, target.l2/2)
        # rotate <target.angle>, counter-clockwise
        x = x1*np.cos(target.angle) - y1*np.sin(target.angle)
        y = x1*np.sin(target.angle) + y1*np.cos(target.angle)

        starting_pos = np.array([x, y, z])
        
        if not IsParallel:
            # Get THETA and PHI range using the geometry of aperture
            # the aperture is considered to be relatively small and usually in the Secend Quadrant
            # if there is a huge change in the geometry of the aperture, this function should be checked
            # only consider Aperture.phi = 90 !!!
            AperCenter = [Aperture1.r*np.sin(Aperture1.theta)*np.cos(Aperture1.phi), Aperture1.r*np.sin(Aperture1.theta)*np.sin(Aperture1.phi), Aperture1.r*np.cos(Aperture1.theta)]
            # Calculate coordinates of 8 vertexes (relative to starting_pos)
            TLF = np.array([AperCenter[0] - Aperture1.l1/2*np.sin(Aperture1.phi) - Aperture1.d/2*np.cos(Aperture1.phi), AperCenter[1] + Aperture1.l1/2*np.cos(Aperture1.phi) - Aperture1.d/2*np.sin(Aperture1.phi), AperCenter[2] + Aperture1.l2/2]) - starting_pos
            TLB = np.array([AperCenter[0] - Aperture1.l1/2*np.sin(Aperture1.phi) + Aperture1.d/2*np.cos(Aperture1.phi), AperCenter[1] + Aperture1.l1/2*np.cos(Aperture1.phi) + Aperture1.d/2*np.sin(Aperture1.phi), AperCenter[2] + Aperture1.l2/2]) - starting_pos
            TRB = np.array([AperCenter[0] + Aperture1.l1/2*np.sin(Aperture1.phi) + Aperture1.d/2*np.cos(Aperture1.phi), AperCenter[1] - Aperture1.l1/2*np.cos(Aperture1.phi) + Aperture1.d/2*np.sin(Aperture1.phi), AperCenter[2] + Aperture1.l2/2]) - starting_pos
            TRF = np.array([AperCenter[0] + Aperture1.l1/2*np.sin(Aperture1.phi) - Aperture1.d/2*np.cos(Aperture1.phi), AperCenter[1] - Aperture1.l1/2*np.cos(Aperture1.phi) - Aperture1.d/2*np.sin(Aperture1.phi), AperCenter[2] + Aperture1.l2/2]) - starting_pos
            BLF = np.array([AperCenter[0] - Aperture1.l1/2*np.sin(Aperture1.phi) - Aperture1.d/2*np.cos(Aperture1.phi), AperCenter[1] + Aperture1.l1/2*np.cos(Aperture1.phi) - Aperture1.d/2*np.sin(Aperture1.phi), AperCenter[2] - Aperture1.l2/2]) - starting_pos
            BLB = np.array([AperCenter[0] - Aperture1.l1/2*np.sin(Aperture1.phi) + Aperture1.d/2*np.cos(Aperture1.phi), AperCenter[1] + Aperture1.l1/2*np.cos(Aperture1.phi) + Aperture1.d/2*np.sin(Aperture1.phi), AperCenter[2] - Aperture1.l2/2]) - starting_pos
            BRB = np.array([AperCenter[0] + Aperture1.l1/2*np.sin(Aperture1.phi) + Aperture1.d/2*np.cos(Aperture1.phi), AperCenter[1] - Aperture1.l1/2*np.cos(Aperture1.phi) + Aperture1.d/2*np.sin(Aperture1.phi), AperCenter[2] - Aperture1.l2/2]) - starting_pos
            BRF = np.array([AperCenter[0] + Aperture1.l1/2*np.sin(Aperture1.phi) - Aperture1.d/2*np.cos(Aperture1.phi), AperCenter[1] - Aperture1.l1/2*np.cos(Aperture1.phi) - Aperture1.d/2*np.sin(Aperture1.phi), AperCenter[2] - Aperture1.l2/2]) - starting_pos
            # Calculate 8 vertexes' angle relative to the starting_pos
            Top4Theta = [np.arccos(TLF[2]/np.linalg.norm(TLF)), np.arccos(TLB[2]/np.linalg.norm(TLB)), np.arccos(TRB[2]/np.linalg.norm(TRB)), np.arccos(TRF[2]/np.linalg.norm(TRF))]
            Bottom4Theta = [np.arccos(BLF[2]/np.linalg.norm(BLF)), np.arccos(BLB[2]/np.linalg.norm(BLB)), np.arccos(BRB[2]/np.linalg.norm(BRB)), np.arccos(BRF[2]/np.linalg.norm(BRF))]
            Right4Phi = [np.arccos(TRB[0]/np.sqrt(TRB[0]**2+TRB[1]**2)), np.arccos(TRF[0]/np.sqrt(TRF[0]**2+TRF[1]**2)), np.arccos(BRB[0]/np.sqrt(BRB[0]**2+BRB[1]**2)), np.arccos(BRF[0]/np.sqrt(BRF[0]**2+BRF[1]**2))]
            Left4Phi = [np.arccos(TLB[0]/np.sqrt(TLB[0]**2+TLB[1]**2)), np.arccos(TLF[0]/np.sqrt(TLF[0]**2+TLF[1]**2)), np.arccos(BLB[0]/np.sqrt(BLB[0]**2+BLB[1]**2)), np.arccos(BLF[0]/np.sqrt(BLF[0]**2+BLF[1]**2))]
            theta_min = max(Top4Theta)
            theta_max = min(Bottom4Theta)
            phi_min = max(Right4Phi)
            phi_max = min(Left4Phi)

            theta = np.random.uniform(theta_min, theta_max)
            phi = np.random.uniform(phi_min, phi_max)
        else: # IsParallel = True, usually for locating the focal plane
            theta = Aperture1.theta
            phi = Aperture1.phi

        # Get starting ENERGY using Ep=En*cos**2(theta)
        proton_direction = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        neutron_direction = np.array([np.sin(Neutron_the*np.pi/180)*np.cos(Neutron_phi*np.pi/180), np.sin(Neutron_the*np.pi/180)*np.sin(Neutron_phi*np.pi/180), np.cos(Neutron_the*np.pi/180)])

        starting_E = Neutron_E*(np.dot(proton_direction, neutron_direction))**2

        # Get ENERGY and POSITION when leave the target
        # need to find energy loss first
        # find the length in target
        starting_E, starting_pos = target.get_Energy_loss(starting_pos, starting_E, theta, phi)
        print(f"Starting Energy: {starting_E:.3f} MeV")
            
        Particle_lst.append(Particle(starting_posX=starting_pos[0], starting_posY=starting_pos[1], starting_posZ=starting_pos[2], starting_E=starting_E, starting_theta=theta*180/np.pi, starting_phi=phi*180/np.pi, particle_name="proton", neutron_E=Neutron_E))

    print(f"Beam initialized with {Nparticle} particles, neutron beam energy: {Neutron_E} MeV.")
    return Particle_lst

def line_segment_intersection(A1, A2, B1, B2):
    """
    Calculate the intersection point of two line segments AB and CD
    
    Parameters:
    A1, A2 - position of particle at t and t+dt
    B1, B2 - end points of line segment 2 (x, y)
    
    Returns:
    The intersection point (x, y) or None (if there is no intersection)
    """
    # Destructure the coordinates of the points
    x1, y1, z1 = A1[0], A1[1], A1[2]
    x2, y2, z2 = A2[0], A2[1], A2[2]
    x3, y3 = B1[0], B1[1]
    x4, y4 = B2[0], B2[1]
    
    # Calculate the denominator
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # Check if the line segments are parallel
    if abs(denom) < 1e-10:
        return None  # The line segments are parallel or coincident
    
    # Calculate the parameters ua and ub
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    # Check if the intersection point is within the line segments
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        # Calculate the coordinates of the intersection point
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        z = (y - y1)/(y2 - y1)*(z2 - z1) + z1
        return x, y, z
    
    return None  # The intersection point is not within the line segment range

# <<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>> #
# ---------------Parameters--------------- #
data_dir = "init_data"
IsTraceSaved = True
# -------------Initialization------------- #
# length: m, angle: degree, energy: MeV
Aperture1 = Aperture(r=0.41, theta=90, phi=120, l1=0.005, l2=0.01, d=0.05)

NPtarget1 = NPtarget(l1=0.005, l2=0.01, d=0.000092, angle=30, target_name="CH2")

FocalPlane1 = FocalPlane(x1 = -0.251, y1 =1.569, x2 = -0.306, y2 = 1.830)

MagField = MagneticField3D(Xshift=-0.3, Yshift=0.52, Zshift=0)
MagField.Read_Bfield2(factor=1.6)

test_particles = Beam_init(Nparticle=100, target=NPtarget1, Neutron_E=14, Neutron_the=90, Neutron_phi=150, IsParallel=True)
# <<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>> #
