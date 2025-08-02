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
import csv

chi1 = 5.3458
c = 299792458
wv = 0.730e-6 # wavelength
chi2 = 1.26e-10 # AlN chi2

# x & y span up to 2 wv
def runSim1(x1, x2, x3, y1, y2, y3, AlNzSpan, DFTz, n, theta=40): 

    V = np.array([
        [x1, y1],
        [x2, y2],
        [x3, y3]
    ])
    
    if(abs(DFTz) > 0.5 * AlNzSpan):
        return 0.0  
    
    # centers most things
    x = 0
    y = 0
    z = 0

    PlaneZ = -0.5 * AlNzSpan - 0.5 * wv 
    print("PlaneZ:", PlaneZ)
    
    SapzSpan = 2e-6 
    Sapz = -0.5 * (AlNzSpan + SapzSpan)
    span = 3e-6 # x & y span for sapphire
    print("SapzSpan:", SapzSpan)
    print("Sapz:", Sapz)
    print("Span:", span)

    FDTDzMin = -0.5 * AlNzSpan - wv 
    FDTDzMax =  AlNzSpan * 0.5 + wv
    FDTDspan = 3 * wv 
    AlNDFTz2 = 0.5 * AlNzSpan + 0.5 * wv
    print("FDTDzMin:", FDTDzMin)
    print("FDTDzMax:", FDTDzMax)
    print("FDTDspan:", FDTDspan)
    print("AlNDFTz2:", AlNDFTz2)

    mesh = 0.1e-6  # only for testing - increase for final runs
    print("Mesh:", mesh)

    fdtd = lumapi.FDTD(hide = True)

    fdtd.eval("q=[2.1297;2.1297;2.1712];setmaterial(addmaterial(\"(n,k) Material\"), \"name\", \"AlN\");setmaterial(\"AlN\", \"Anisotropy\", 1);setmaterial(\"AlN\", \"Refractive Index\", q);")

    fdtd.addfdtd()
    fdtd.addmesh()

    fdtd.addplane()
    fdtd.set("name", "PlaneWave")

    fdtd.addtriangle()
    fdtd.set("name", "AlNfilm")
    fdtd.set("material", "AlN")

    fdtd.addrect()
    fdtd.set("name", "Sapphire")
    fdtd.set("material", "Al2O3 - Palik")

    meshxspan = np.max(np.abs(V[:,0])) * 2
    meshyspan = np.max(np.abs(V[:,1])) * 2

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
                    ("vertices", V),
                    ("z span", AlNzSpan))),

        ("mesh", (
                    ("dx", mesh),
                    ("dy", mesh),
                    ("x", x),
                    ("y", y),
                    ("z", z),
                    ("x span", meshxspan),
                    ("y span", meshyspan),
                    ("z span", AlNzSpan))),

        ("FDTD", (
                    ("x", x),
                    ("y", y),
                    ("z", z),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan),
                    ("z min", FDTDzMin),
                    ("z max", FDTDzMax),
                    ("x min bc", "bloch"), # make bloch later
                    ("y min bc", "bloch"))), # make bloch later

        ("Sapphire", (
                    ("x", x),
                    ("y", y),
                    ("z", Sapz),
                    ("x span", span),
                    ("y span", span),
                    ("z span", SapzSpan),
                    ("material", "Al2O3 - Palik")))
    )

    for obj, parameters in configuration:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)

    for i in range(n):
        fdtd.adddftmonitor()
        fdtd.set("name", f"AlNDFT{i+1}")
        fdtd.setnamed(f"AlNDFT{i+1}", "x", x)
        fdtd.setnamed(f"AlNDFT{i+1}", "y", y)
        fdtd.setnamed(f"AlNDFT{i+1}", "z", DFTz + (-1 ** i ) * 0.005e-6 * (i+2 // 2))
        fdtd.setnamed(f"AlNDFT{i+1}", "x span", FDTDspan)
        fdtd.setnamed(f"AlNDFT{i+1}", "y span", FDTDspan)

    fdtd.save("triangle-test")
    fdtd.run()

    for i in range(n):
        fdtd.eval(f"E{i+1} = rectilineardataset(\"EM Fields\", getresult(\"AlNDFT{i+1}\", \"x\"), getresult(\"AlNDFT{i+1}\", \"y\"), getresult(\"AlNDFT{i+1}\", \"z\"));")
        fdtd.eval("chi1 = 5.3458;")
        fdtd.eval(f"Ex{i+1}= getresult(\"AlNDFT{i+1}\", \"Ex\");Ey{i+1}= getresult(\"AlNDFT{i+1}\", \"Ey\");Ez{i+1}= getresult(\"AlNDFT{i+1}\", \"Ez\");")
        fdtd.eval(f"E{i+1}x = (2 * {chi2} * Ez{i+1} * Ex{i+1}) / chi1; E{i+1}y = (2 * {chi2} * Ez{i+1} * Ey{i+1}) / chi1; E{i+1}z = ({chi2} * (Ex{i+1} ^ 2 + Ey{i+1} ^ 2) - {chi2} * Ez{i+1} ^ 2) / chi1;")
        fdtd.eval(f"E{i+1}.addparameter(\"lambda\", 299792458/getresult(\"AlNDFT{i+1}\", \"f\"), \"f\", getresult(\"AlNDFT{i+1}\", \"f\"));")
        fdtd.eval(f"E{i+1}.addattribute(\"E\", E{i+1}x, E{i+1}y, E{i+1}z);")

    fdtd.switchtolayout()
    fdtd.select("PlaneWave")
    fdtd.delete()

    fdtd.adddftmonitor()
    fdtd.set("name", "AlNDFTair")

    for i in range(n):
        fdtd.eval(f"addimportedsource; importdataset(E{i+1});set(\"name\", \"source{i+1}\");set(\"x\", {x});set(\"y\", {y});set(\"z\", {z});set(\"injection axis\", \"z\");set(\"direction\", \"forward\");")

    configuration2 = (
        ("AlNDFTair", (
                    ("x", x),
                    ("y", y),
                    ("z", AlNDFTz2),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan))),
    )

    for obj, parameters in configuration2:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)

    fdtd.save("triangle-test2")
    fdtd.run()

    result = fdtd.getresult("AlNDFTair", "power")

    return real(result)[0][0] / 3.7e8

x1 = -7.3e-07
x2 = 2.632116472180357e-07
x3 = -1.350505637274167e-07
y1 = 7.3e-07
y2 = -1.0467059205885162e-07
y3 = -4.300044395634675e-07
AlNzSpan = 1.46e-06
DFTz = -2.2357263480853136e-09
theta = 30


power = runSim1(x1, x2, x3, y1, y2, y3, AlNzSpan, DFTz, 1, theta=theta)
print(f"1 QW: {power}")

power = runSim1(x1, x2, x3, y1, y2, y3, AlNzSpan, DFTz, 10, theta=theta)
print(f"10 QW: {power}")
