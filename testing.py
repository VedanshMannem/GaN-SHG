import os
import sys

sys.path.append("C:\\Program Files\\Lumerical\\v251\\api\\python\\") 
sys.path.append(os.path.dirname(__file__)) 
import lumapi
import numpy as np

def run_topology_optimization(nx=50, ny=50, x_span=10e-6, y_span=10e-6, AlNzSpan=2e-6):
    fdtd = lumapi.FDTD(hide=True)

    ## -------------------------
    ## Setup Materials & Geometry
    ## -------------------------

    # Define Sapphire substrate
    fdtd.addrect()
    fdtd.set("name", "Sapphire")
    fdtd.set("material", "Al2O3 - Palik")
    fdtd.set("x span", x_span)
    fdtd.set("y span", y_span)
    fdtd.set("z span", 2e-6)
    fdtd.set("z", -0.5 * (AlNzSpan + 2e-6))

    # Add Optimizable Geometry (AlN/air pixels)
    fdtd.addstructure("optimizable geometry")
    fdtd.set("name", "TopoRegion")
    fdtd.set("material 1", "AlN")
    fdtd.set("material 2", "Air")

    fdtd.set("x span", x_span)
    fdtd.set("y span", y_span)
    fdtd.set("z span", AlNzSpan)

    fdtd.set("nx", nx)
    fdtd.set("ny", ny)

    # Optional: Apply filtering to reduce checkerboarding
    fdtd.set("filter radius", 0.2e-6)
    fdtd.set("projection beta", 2)

    ## -------------------------
    ## Setup Source & Monitors
    ## -------------------------

    # Add fundamental source (1064 nm plane wave)
    fdtd.addplane()
    fdtd.set("name", "InputSource")
    fdtd.set("injection axis", "z")
    fdtd.set("direction", "forward")
    fdtd.set("x span", x_span)
    fdtd.set("y span", y_span)
    fdtd.set("z", -0.5 * AlNzSpan - 0.5e-6)
    fdtd.set("wavelength start", 1.064e-6)
    fdtd.set("wavelength stop", 1.064e-6)

    # Add output monitor at SHG plane (proxy for SHG intensity)
    fdtd.addpower()
    fdtd.set("name", "OutputMonitor")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x span", x_span)
    fdtd.set("y span", y_span)
    fdtd.set("z", 0.5 * AlNzSpan + 0.5e-6)

    ## -------------------------
    ## Setup FDTD Region
    ## -------------------------

    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    fdtd.set("x span", x_span)
    fdtd.set("y span", y_span)
    fdtd.set("z span", AlNzSpan + 2e-6)

    fdtd.set("x min bc", "periodic")
    fdtd.set("x max bc", "periodic")
    fdtd.set("y min bc", "periodic")
    fdtd.set("y max bc", "periodic")
    fdtd.set("z min bc", "PML")
    fdtd.set("z max bc", "PML")

    fdtd.set("mesh accuracy", 2)  # Use higher accuracy for final runs

    ## -------------------------
    ## Setup Figure of Merit (FOM)
    ## -------------------------

    fdtd.addfigureofmerit("power")
    fdtd.set("name", "SHG_Power_FOM")
    fdtd.set("monitor", "OutputMonitor")
    fdtd.set("optimization type", "maximize")

    ## -------------------------
    ## Setup Optimization
    ## -------------------------

    fdtd.addoptimization()
    fdtd.set("name", "TopoOpt")
    fdtd.set("optimization variables", ["TopoRegion.alpha"])
    fdtd.set("figure of merit", ["SHG_Power_FOM"])

    # Use Method of Moving Asymptotes (MMA) for topology optimization
    fdtd.set("optimization algorithm", "Method of Moving Asymptotes (MMA)")
    fdtd.set("max iterations", 50)
    fdtd.set("tolerance", 1e-3)

    ## -------------------------
    ## Run Optimization
    ## -------------------------

    print("Starting topology optimization...")

    fdtd.runoptimization("TopoOpt")

    ## -------------------------
    ## Get Results
    ## -------------------------

    alpha_final = fdtd.getnamed("TopoRegion", "alpha")
    np.save("optimized_alpha.npy", alpha_final)

    print("Topology optimization complete.")
    print("Final alpha saved to optimized_alpha.npy")

    fdtd.save("topology_optimized_design.fsp")

    fdtd.close()

if __name__ == "__main__":
    run_topology_optimization()
