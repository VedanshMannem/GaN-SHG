import sys, os
from numpy import real
sys.path.append("C:\\Program Files\\Lumerical\\v251\\api\\python\\") 
sys.path.append(os.path.dirname(__file__)) 
import lumapi
import numpy as np

wv = 1.064e-6
chi1 = 5.3458
theta = 30
c = 299792458
chi2 = 1.26e-10

GaNx = 0
GaNy = 0
GaNz = 0

span = 2 * wv
GaNzSpan = wv

PlaneZ = GaNz - 0.5 * GaNzSpan - wv
PlaneX = 0
PlaneY = 0

Sapx = 0
Sapy = 0
SapSpan = 2 * wv
Sapz = -0.5 * (GaNzSpan + SapSpan)

FDTDx = 0
FDTDy = 0
FDTDzMin = Sapz - wv
FDTDzMax = GaNz + GaNzSpan * 0.5 + wv
FDTDspan = span

mesh = 0.085e-6

GaNDFTx = 0
GaNDFTy = 0
GaNDFTz_top = GaNz + 0.5 * GaNzSpan + wv
GaNDFTz_bottom = Sapz - wv

def runSim1():
    fdtd = lumapi.FDTD(hide=False)

    fdtd.eval("mat = addmaterial(\"Chi2\");setmaterial(mat, \"name\", \"GaN_chi2\");setmaterial(\"GaN_chi2\", \"Chi1\", 5.3458);setmaterial(\"GaN_chi2\", \"n_o\", 2.3333);setmaterial(\"GaN_chi2\", \"n_e\", 2.3095);setmaterial(\"GaN_chi2\", \"Chi2\", 1.26e-10);")

    fdtd.addfdtd()
    fdtd.addmesh()

    fdtd.addplane()
    fdtd.set("name", "PlaneWave")

    fdtd.addrect()
    fdtd.set("name", "GaNfilm")
    fdtd.set("material", "GaN_chi2")

    fdtd.addrect()
    fdtd.set("name", "Sapphire")
    fdtd.set("material", "Al2O3 - Palik")

    fdtd.adddftmonitor()
    fdtd.set("name", "GaNDFT")

    configuration = (
        ("PlaneWave", (
            ("x", PlaneX),
            ("y", PlaneY),
            ("z", PlaneZ),
            ("x span", FDTDspan),
            ("y span", FDTDspan),
            ("angle theta", theta),
            ("amplitude", 1),
            ("wavelength start", wv),
            ("wavelength stop", wv))),
        ("GaNfilm", (
            ("x", GaNx),
            ("y", GaNy),
            ("z", GaNz),
            ("z span", GaNzSpan),
            ("x span", span),
            ("y span", span))),
        ("mesh", (
            ("dx", mesh),
            ("dy", mesh),
            ("x", GaNx),
            ("y", GaNy),
            ("z", GaNz),
            ("z span", GaNzSpan),
            ("structure", "GaNfilm"),
            ("x span", span),
            ("y span", span))),
        ("FDTD", (
            ("x", FDTDx),
            ("y", FDTDy),
            ("x span", FDTDspan),
            ("y span", FDTDspan),
            ("z min", FDTDzMin),
            ("z max", FDTDzMax),
            ("x min bc", "PML"),
            ("y min bc", "PML"))),
        ("Sapphire", (
            ("x", Sapx),
            ("y", Sapy),
            ("z", Sapz),
            ("x span", SapSpan),
            ("y span", SapSpan),
            ("z span", SapSpan),
            ("material", "Al2O3 - Palik"))),
        ("GaNDFT", (
            ("x", GaNDFTx),
            ("y", GaNDFTy),
            ("z", 0),
            ("x span", FDTDspan),
            ("y span", FDTDspan)))
    )

    for obj, parameters in configuration:
        for k, v in parameters:
            fdtd.setnamed(obj, k, v)

    fdtd.adddftmonitor()
    fdtd.set("name", "SH_top")
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("z", GaNDFTz_top)
    fdtd.set("x span", FDTDspan)
    fdtd.set("y span", FDTDspan)
    fdtd.set("frequency start", 2*c/wv)
    fdtd.set("frequency stop", 2*c/wv)

    fdtd.adddftmonitor()
    fdtd.set("name", "SH_bottom")
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("z", GaNDFTz_bottom)
    fdtd.set("x span", FDTDspan)
    fdtd.set("y span", FDTDspan)
    fdtd.set("frequency start", 2*c/wv)
    fdtd.set("frequency stop", 2*c/wv)

    fdtd.save("GaN_chi2_SHG")
    fdtd.run()

    P_top = fdtd.getresult("SH_top", "power")
    P_bottom = fdtd.getresult("SH_bottom", "power")

    print(P_top)
    print(P_bottom)

    return real(P_top)[0][0], real(P_bottom)[0][0]

power_top, power_bottom = runSim1()
print(f"Top SH power: {power_top}, Bottom SH power: {power_bottom}")
