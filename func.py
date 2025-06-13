import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import physical_constants

data_dir = "init_data"

IsTraceSaved = True

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
    
class Particle:
    def __init__(self, starting_posX, starting_posY, starting_posZ, starting_velX, starting_velY, starting_velZ, particle_name):
        self.pos = np.array([starting_posX, starting_posY, starting_posZ])
        self.vel = np.array([starting_velX, starting_velY, starting_velZ])
        self.tof = 0

        self.name = particle_name
        if IsTraceSaved:
            self.trace = [[self.pos[0], self.pos[1], self.pos[2], self.tof]]
        else:
            self.trace = None

        if particle_name == "proton":
            self.m = physical_constants["proton mass"][0] # [kg]
            self.q = physical_constants["elementary charge"][0] # [C]
        elif particle_name == "alpha":
            self.m = 4*physical_constants["proton mass"][0] # [kg]
            self.q = 2*physical_constants["elementary charge"][0] # [C]
        else:
            raise ValueError("Invalid particle name")
        
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
            self.trace.append([self.pos[0], self.pos[1], self.pos[2], self.tof])
        else:
            raise ValueError("You cannot record position if IsTraceSaved is False!")