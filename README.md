# Synthetic Magnetic Proton Recoil (MPR) Simulator

**Author**: Xutao Xu  
**Date**: June 18, 2025  

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/Physics-Nuclear%20Spectroscopy-green" alt="Physics">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</div>

## Overview
This advanced Python simulation models charged particle behavior in magnetic spectrometer systems for neutron spectroscopy applications. The v2.0 update introduces significant enhancements:

1. **Focal Plane Detection** - Records particle strikes on the detector plane
2. **Precision Intersection Calculation** - Accurately tracks particle-plane collisions
3. **Parallel Beam Generation** - New option for focal plane calibration
4. **Magnetic Field Scaling** - Flexible adjustment of field strength

The simulation is essential for designing and optimizing Magnetic Proton Recoil (MPR) spectrometers used in nuclear physics experiments.

## Key Features

### 1. Enhanced Particle Physics Modeling
```python
class Particle:
    # Tracks position, velocity, and time-of-flight
    # Implements Boris push algorithm with collision detection
    # Records full trajectories when enabled
```

### 2. Intelligent Geometry Components
```python
# Aperture with configurable position and dimensions
Aperture1 = Aperture(r=0.41, theta=90, phi=120, l1=0.005, l2=0.01, d=0.05)

# Target with material properties and rotation
NPtarget1 = NPtarget(l1=0.005, l2=0.01, d=0.000092, angle=30, target_name="CH2")

# Focal plane with strike recording
FocalPlane1 = FocalPlane(x1=-0.251, y1=1.569, x2=-0.306, y2=1.830)
```

### 3. Advanced Magnetic Field Handling
```python
# Magnetic field with coordinate transformation
MagField = MagneticField3D(Xshift=-0.3, Yshift=0.52, Zshift=0)
MagField.Read_Bfield2(factor=1.6)  # Field strength scaling
```

### 4. Precision Collision Detection
```python
def line_segment_intersection(A1, A2, B1, B2):
    # Calculates 3D intersection points
    # Handles edge cases and parallel segments
```

## Key Components

### 1. Particle Class (`Particle`)
- Tracks position, velocity, and time-of-flight
- Implements Boris push algorithm with collision detection
- Stores incident neutron energy for spectroscopy
- Records full trajectories when enabled

### 2. Target Class (`NPtarget`)
- Models energy loss in materials using stopping power data
- Calculates exit points and energies with rotation
- Handles CH2 targets with configurable dimensions

### 3. Magnetic Field Class (`MagneticField3D`)
- Loads field data from multiple formats (MATLAB, text)
- Provides 3D interpolation of field vectors
- Enables coordinate transformation and field scaling

### 4. Focal Plane Class (`FocalPlane`)
- Defines detector plane geometry
- Records strike positions with incident neutron energies
- Outputs results to file for analysis

### 5. Beam Initialization (`Beam_init`)
- Generates particles with proper kinematics
- Angular spread or parallel beam options
- Simulates energy loss in targets

## Configuration Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `data_dir` | Input data directory | `"init_data"` |
| `IsTraceSaved` | Enable trajectory recording | `True` |
| `Aperture` | Collimator geometry | r=0.41m, l1=5mm, l2=10mm |
| `NPtarget` | Target specifications | l1=5mm, l2=10mm, d=92μm |
| `FocalPlane` | Detector plane endpoints | (-0.251,1.569) to (-0.306,1.830) |

## Usage Example

```python
# Initialize components
Aperture1 = Aperture(r=0.41, theta=90, phi=120, l1=0.005, l2=0.01, d=0.05)
NPtarget1 = NPtarget(l1=0.005, l2=0.01, d=0.000092, angle=30, target_name="CH2")
FocalPlane1 = FocalPlane(x1=-0.251, y1=1.569, x2=-0.306, y2=1.830)
MagField = MagneticField3D(Xshift=-0.3, Yshift=0.52, Zshift=0)
MagField.Read_Bfield2(factor=1.6)

# Generate parallel beam for focal plane calibration
test_particles = Beam_init(Nparticle=100, target=NPtarget1, 
                           Neutron_E=14, Neutron_the=90, 
                           Neutron_phi=150, IsParallel=True)

# Simulation parameters
dt = 1e-12
t_max = 1e-8
t_now = 0

# Main simulation loop
while t_now < t_max:
    t_now += dt
    for particle in test_particles:
        B = MagField.get_Bvector_at(*particle.pos)
        particle.Boris_push(B, dt)
        
# Save results
FocalPlane1.write_record("focal_plane_strikes.txt", test_particles)
print(f"Simulation complete! {len(FocalPlane1.record)} particles detected.")
```

## Data Requirements
1. **Stopping Power Data** (`ESP.dat`):
   - Energy loss profiles for materials
   
2. **Magnetic Field Data**:
   - Text format (`combined_field_upgrade.txt`)
   - MATLAB format (`BposX.mat`, `BposY.mat`, `BposZ.mat`)
   
3. **Optional**:
   - Custom field configurations via `Arbitrary_Bfield()`

## Applications
- Neutron energy spectroscopy
- Radiation detector design
- Particle spectrometer optimization
- Nuclear fusion diagnostics
- Accelerator physics experiments

## Getting Started
1. Install dependencies:
   ```bash
   pip install numpy scipy
   ```

2. Configure parameters at bottom of `func.py`

3. Run simulation:
   ```bash
   python func.py
   ```

4. Analyze output in `focal_plane_strikes.txt`

## Physics Model
The simulation implements:
1. **n-p scattering kinematics**:
   $$E_p = E_n \cos^2\theta$$
2. **Energy loss in materials**:
   $$\frac{dE}{dx} = f_{\text{material}}(E)$$
3. **Boris push algorithm**:
   $$\vec{v}_{n+1} = M^{-1} \vec{v}_n$$
4. **Magnetic Lorentz force**:
   $$\vec{F} = q(\vec{v} \times \vec{B})$$
