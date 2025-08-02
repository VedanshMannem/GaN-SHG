import sys, os

from numpy import delete, real, shape
sys.path.append("C:\\Program Files\\Lumerical\\v251\\api\\python\\") 
sys.path.append(os.path.dirname(__file__)) 

from pprint import pprint
import lumapi # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

chi1 = 5.3458
c = 299792458
wv = 0.73e-6  # wavelength
chi2 = 1.26e-10 

# x & y span up to 2 wv
def runSim1(AlNxSpan, AlNySpan, AlNzSpan, DFTz=0, theta=40): 

    x= 0
    y= 0
    z= 0

    if(abs(DFTz) > 0.5 * AlNzSpan):
        return 0.0  

    PlaneZ = -0.5 * AlNzSpan - 0.5 * wv
    print("PlaneZ:", PlaneZ)
    
    SapzSpan = 2 * wv
    Sapz = z - 0.5 * (AlNzSpan + SapzSpan)
    span = 2.5 * wv  # x & y span for sapphire

    print("Sapz:", Sapz)
    print("SapzSpan:", SapzSpan)

    FDTDzMin = -0.5 * AlNzSpan -  wv
    FDTDzMax =  AlNzSpan * 0.5 +  wv
    FDTDspan = 2.5 * wv
    AlNDFTz2 = 0.5 * AlNzSpan + 0.5 * wv

    print("FDTDzMin:", FDTDzMin)
    print("FDTDzMax:", FDTDzMax)
    print("FDTDspan:", FDTDspan)
    print("AlNDFTz2:", AlNDFTz2)

    mesh = 0.1e-6  # only for testing - increase for final runs
    print("Mesh:", mesh)

    fdtd = lumapi.FDTD(hide = False)

    fdtd.eval("q=[2.1297;2.1297;2.1712];setmaterial(addmaterial(\"(n,k) Material\"), \"name\", \"AlN\");setmaterial(\"AlN\", \"Anisotropy\", 1);setmaterial(\"AlN\", \"Refractive Index\", q);")

    fdtd.addfdtd()
    fdtd.addmesh()

    fdtd.addplane()
    fdtd.set("name", "PlaneWave")

    fdtd.addrect()
    fdtd.set("name", "AlNfilm")
    fdtd.set("material", "AlN")

    fdtd.addrect()
    fdtd.set("name", "Sapphire")
    fdtd.set("material", "Al2O3 - Palik")

    fdtd.adddftmonitor()
    fdtd.set("name", "AlNDFT") # AlNDFT in Lumerical

    configuration = (
        ("PlaneWave", (
                    ("x", x),
                    ("y", y),
                    ("z", PlaneZ),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan),
                    ("amplitude", 3.7e8),
                    ("angle theta", theta),
                    ("wavelength start", wv),
                    ("wavelength stop", wv))),

        ("AlNfilm", (
                    ("x", x),
                    ("y", y),
                    ("z", z),
                    ("x span", AlNxSpan),
                    ("y span", AlNySpan),
                    ("z span", AlNzSpan))),

        ("mesh", (
                    ("dx", mesh),
                    ("dy", mesh),
                    ("x", x),
                    ("y", y),
                    ("z", z),
                    ("x span", AlNxSpan),
                    ("y span", AlNySpan),
                    ("z span", AlNzSpan))),

        ("FDTD", (
                    ("x", x),
                    ("y", y),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan),
                    ("z min", FDTDzMin),
                    ("z max", FDTDzMax),
                    ("x min bc", "bloch"),  # switch to bloch later
                    ("y min bc", "bloch"))), # switch to bloch later

        ("Sapphire", (
                    ("x", x),
                    ("y", y),
                    ("z", Sapz),
                    ("x span", span),
                    ("y span", span),
                    ("z span", SapzSpan),
                    ("material", "Al2O3 - Palik"))),

        ("AlNDFT", (
                    ("x", x),
                    ("y", y),
                    ("z", DFTz),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan)))
    )

    for obj, parameters in configuration:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)


    fdtd.save("rect-test")
    fdtd.run()
    
    # Full eval script to get the imported source
    fdtd.eval("E2 = rectilineardataset(\"EM Fields\", getresult(\"AlNDFT\", \"x\"), getresult(\"AlNDFT\", \"y\"), getresult(\"AlNDFT\", \"z\"));")
    fdtd.eval("chi1 = 5.3458;")
    fdtd.eval("Ex= getresult(\"AlNDFT\", \"Ex\");Ey= getresult(\"AlNDFT\", \"Ey\");Ez= getresult(\"AlNDFT\", \"Ez\");")
    fdtd.eval(f"E2x = (2 * {chi2} * Ez * Ex) / chi1; E2y = (2 * {chi2} * Ez * Ey) / chi1; E2z = ({chi2} * (Ex ^ 2 + Ey ^ 2) - {chi2} *  Ez ^ 2) / chi1;")
    fdtd.eval("E2.addparameter(\"lambda\", 299792458/getresult(\"AlNDFT\", \"f\"), \"f\", getresult(\"AlNDFT\", \"f\"));")
    fdtd.eval("E2.addattribute(\"E\", E2x, E2y, E2z);")
    
    fdtd.switchtolayout()
    fdtd.select("PlaneWave")
    fdtd.delete()

    fdtd.adddftmonitor()
    fdtd.set("name", "AlNDFT2")

    fdtd.eval(f"addimportedsource; importdataset(E2);set(\"name\", \"source2\");set(\"x\", {x});set(\"y\", {y});set(\"z\", {DFTz});set(\"injection axis\", \"z\");set(\"direction\", \"forward\");")

    configuration2 = (
        ("AlNDFT2", (
                    ("x", x),
                    ("y", y),
                    ("z", AlNDFTz2),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan))),
    )

    for obj, parameters in configuration2:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)

    fdtd.save("rect-test2")
    fdtd.run()

    result = fdtd.getresult("AlNDFT2", "power")

    return real(result)[0][0] / 3.7e8 # for raw transmission - BOp

    # return real(result)[0][0] / (3.7e8) # for actual transmission

# 37,0.345,1.46,1.355,-0.144,25.5,2.0419825923541257,success
print(runSim1(0.345e-6, 1.46e-6, 1.355e-6, -0.144e-6, 25.5))