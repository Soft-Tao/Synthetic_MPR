import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import physical_constants

data_dir = "init_data"

IsTraceSaved = True

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

Aperture1 = Aperture(0.4, 90, 120, 0.005, 0.001, 0.01)

class MagneticField3D:
    def __init__(self, Xshift, Yshift, Zshift): 
        # Cooridinates of MagFieldGrid's origin in the new coordinate system (Xshift, Yshift, Zshift) [m]
        self.xgrid = sio.loadmat(data_dir + "/BposX.mat")["B_posX"][:,0]/1000 + Xshift # [m]
        self.ygrid = sio.loadmat(data_dir + "/BposY.mat")["B_posY"][:,0]/1000 + Yshift # [m]
        self.zgrid = sio.loadmat(data_dir + "/BposZ.mat")["B_posZ"][:,0]/1000 + Zshift # [m]

        self.Nx = len(self.xgrid)
        self.Ny = len(self.ygrid)
        self.Nz = len(self.zgrid)
        print("Spatial Grid: Nx, Ny, Nz = ", self.Nx, self.Ny, self.Nz)

        self.Bx = sio.loadmat(data_dir + "/Bxfield.mat")["Bf_X"].reshape((self.Nx, self.Ny, self.Nz)) # [T]
        self.By = sio.loadmat(data_dir + "/Byfield.mat")["Bf_Y"].reshape((self.Nx, self.Ny, self.Nz)) # [T]
        self.Bz = sio.loadmat(data_dir + "/Bzfield.mat")["Bf_Z"].reshape((self.Nx, self.Ny, self.Nz)) # [T]

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
    
    # This fuction is mainly used for testing
    # Loading arbitrary Bfield with xgrid, ygrid, zgrid [Narray of size Nx, Ny and Nz] and Bx, By, Bz [Narray of size (Nx, Ny, Nz)]
    def reset_field(self, xgrid, ygrid, zgrid, Bx, By, Bz):
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.zgrid = zgrid
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
    
class Particle:
    # starting_E in [MeV], (E, theta [degree], phi [degree]) to (vx, vy, vz) [m/s]
    def __init__(self, starting_posX, starting_posY, starting_posZ, starting_E, starting_theta, starting_phi, particle_name):
        self.pos = np.array([starting_posX, starting_posY, starting_posZ])
        self.tof = 0

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
        A = self.q*dt/(2*self.m)
        # Only magnetic field is considered, to form the coefficient matrix:
        M = np.array([[1, -A*BfieldVec[2], A*BfieldVec[1]], [A*BfieldVec[2], 1, -A*BfieldVec[0]], [-A*BfieldVec[1], A*BfieldVec[0], 1]])

        # v_{n+1} = M^{-1} \cdot v_{n}
        self.vel = np.matmul(np.linalg.inv(M), self.vel)
        # r_{n+1} = r_{n} + v_{n+1} \cdot dt
        self.pos = self.pos + self.vel*dt
        self.tof += dt
    
    def record_pos(self):
        if IsTraceSaved:
            self.trace.append([self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.tof])
        else:
            raise ValueError("You cannot record position if IsTraceSaved is False!")

# This version neglects n-p target's size, all the protons come from the point (0, 0, 0) with a specific energy
def Beam_init(Nparticle):

    Particle_lst = []

    starting_pos = np.array([0, 0, 0])
    starting_E = 5
    
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

    for i in range(Nparticle):
        theta = np.random.uniform(theta_min, theta_max)
        phi = np.random.uniform(phi_min, phi_max)
        
        Particle_lst.append(Particle(starting_posX=0, starting_posY=0, starting_posZ=0, starting_E=starting_E, starting_theta=theta*180/np.pi, starting_phi=phi*180/np.pi, particle_name="proton"))

    return Particle_lst

