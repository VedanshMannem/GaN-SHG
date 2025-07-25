import sys, os

from numpy import delete, real, shape
from scipy import integrate
sys.path.append("C:\\Program Files\\Lumerical\\v251\\api\\python\\") 
sys.path.append(os.path.dirname(__file__)) 

from pprint import pprint
import lumapi # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

# Optimize for power at the 2nd monitor not T

chi1 = 5.3458
theta = 40 # angle of plane wave
c = 299792458
chi2 = 1.26 * 10 ** -12 # quartz chi2

GaNx = 0
GaNy = 0
GaNz = 0

span = 1.064e-6
GaNzSpan =  1.064e-6

PlaneZ = GaNz - 0.5 * GaNzSpan - 1.064e-6 # record
PlaneX = 0
PlaneY = 0

Sapx = 0
Sapy = 0
SapSpan = 2.128e-6 # record
Sapz = -0.5 * (GaNzSpan + SapSpan)

FDTDx = 0
FDTDy = 0
FDTDzMin = Sapz - 1.064e-6
FDTDzMax = GaNz + GaNzSpan * 0.5 + 1.064e-6
FDTDspan =  span + 1.064e-6 # x & y span

mesh = 0.085e-6  

# Center of the GaN
GaNDFTx = 0
GaNDFTy = 0
GaNDFTz = 0 

GaNDFTx2 = 0
GaNDFTy2 = 0
GaNDFTz2 = GaNz + 0.5 * GaNzSpan + 1.064e-6

def runSim1():

    fdtd = lumapi.FDTD(hide = False)

    # AlN indices
    fdtd.eval("q=[2.1297;2.1297;2.1712];setmaterial(addmaterial(\"(n,k) Material\"), \"name\", \"GaN\");setmaterial(\"GaN\", \"Anisotropy\", 1);setmaterial(\"GaN\", \"Refractive Index\", q);")

    fdtd.addfdtd()
    fdtd.addmesh()

    fdtd.addplane()
    fdtd.set("name", "PlaneWave")

    fdtd.addrect()
    fdtd.set("name", "GaNfilm")
    fdtd.set("material", "GaN")

    fdtd.addrect()
    fdtd.set("name", "Sapphire")
    fdtd.set("material", "Al2O3 - Palik")

    fdtd.adddftmonitor()
    fdtd.set("name", "GaNDFT") # GaNDFT in Lumerical

    # material = GaN
    # rect structure = GaNfilm

    configuration = (
        ("PlaneWave", (
                    ("x", PlaneX),
                    ("y", PlaneY),
                    ("z", PlaneZ),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan),
                    ("angle theta", theta),
                    ("amplitude", 3.7e8),
                    ("wavelength start", 1.064e-6),
                    ("wavelength stop", 1.064e-6))),

        ("GaNfilm", (
                    ("x",GaNx),
                    ("y",GaNy),
                    ("z",GaNz),
                    ("z span", GaNzSpan),
                    ("x span", span),
                    ("y span", span))),

        ("mesh", (("dx", mesh),
                  ("dy", mesh),
                  ("x", GaNx),
                  ("y", GaNy),
                  ("z", GaNz),
                  ("z span", GaNzSpan),
                  ("structure", "GaNfilm"),
                  ("x span", span),
                  ("y span", span))),

        ("FDTD", (("x",FDTDx),
                  ("y",FDTDy),
                  ("x span", FDTDspan),
                  ("y span", FDTDspan),
                  ("z min", FDTDzMin),
                  ("z max", FDTDzMax),
                  ("x min bc", "periodic"),
                  ("y min bc", "periodic"))),

        ("Sapphire", ( 
                  ("x",Sapx),
                  ("y",Sapy),
                  ("z",Sapz),
                  ("x span", SapSpan),
                  ("y span", SapSpan),
                  ("z span", SapSpan),
                  ("material", "Al2O3 - Palik"))),

        ("GaNDFT", (
                    ("x", GaNDFTx),
                    ("y", GaNDFTy),
                    ("z", GaNDFTz),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan)))
    )

    for obj, parameters in configuration:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)


    fdtd.save("slab-test")
    fdtd.run()
    
    # Full eval script to get the imported source
    fdtd.eval("E2 = rectilineardataset(\"EM Fields\", getresult(\"GaNDFT\", \"x\"), getresult(\"GaNDFT\", \"y\"), getresult(\"GaNDFT\", \"z\"));")
    fdtd.eval("chi1 = 5.3458;")
    fdtd.eval(f"Ex= getresult(\"GaNDFT\", \"Ex\");Ey= getresult(\"GaNDFT\", \"Ey\");Ez= getresult(\"GaNDFT\", \"Ez\");")
    fdtd.eval(f"E2x = (2 * 11.33 * {chi2} * Ez * Ex) / chi1; E2y = (2 * 11.33 * {chi2} * Ez * Ey) / chi1; E2z = (11.33 * {chi2} * (Ex ^ 2 + Ey ^ 2) - 22.66 * {chi2} * Ez ^ 2) / chi1;")
    fdtd.eval("E2.addparameter(\"lambda\", 299792458/getresult(\"GaNDFT\", \"f\"), \"f\", getresult(\"GaNDFT\", \"f\"));")
    fdtd.eval("E2.addattribute(\"E\", E2x, E2y, E2z);")
    
    fdtd.switchtolayout()
    fdtd.select("PlaneWave")
    fdtd.delete()

    fdtd.adddftmonitor()
    fdtd.set("name", "GaNDFT2")

    fdtd.eval(f"addimportedsource; importdataset(E2);set(\"name\", \"source2\");set(\"x\", {GaNDFTx});set(\"y\", {GaNDFTy});set(\"z\", {GaNDFTz});set(\"injection axis\", \"z\");set(\"direction\", \"forward\");")

    configuration2 = (
        ("GaNDFT2", (
                    ("x", GaNDFTx2),
                    ("y", GaNDFTy2),
                    ("z", GaNDFTz2),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan))),
    )

    for obj, parameters in configuration2:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)

    fdtd.save("circle-test2")
    fdtd.run()

    result = fdtd.getresult("GaNDFT2", "power")

    return real(result)[0][0]

power = runSim1()
print(f"Parameters used: \n span: {span} \n GaNzSpan: {GaNzSpan} \n mesh: {mesh} \n Sapphire: {SapSpan} \n everything else: 1.064e-6")
print("Power at 2nd monitor:", power)