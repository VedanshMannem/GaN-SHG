import sys, os
import csv

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
wv = 1.064e-6 # wavelength
chi2 = 1.26e-10 # quartz chi2


def log_to_csv(x, y, z, AlNxSpan, AlNySpan, AlNzSpan, DFTz, theta, n, power):
    

    with open("./RectTesting/mqw_data.csv", "a", newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow([
            x, 
            y, 
            z, 
            AlNxSpan,
            AlNySpan,
            AlNzSpan,
            DFTz,
            theta,
            n,
            power
        ])

# x & y span up to 2 wv
def runSim1(x, y, z, AlNxSpan, AlNySpan, AlNzSpan, n, DFTz=0, theta=40): 

    if(abs(DFTz) > 0.5 * AlNzSpan):
        return 0.0
    
    print("x:", x, "y:", y, "z:", z)
    print("AlNxSpan:", AlNxSpan, "AlNySpan:", AlNySpan, "AlNzSpan:", AlNzSpan)

    PlaneZ = -0.5 * AlNzSpan - 0.532e-6
    print("PlaneZ:", PlaneZ)
    
    SapzSpan = 4e-6
    print("SapzSpan:", SapzSpan)
    Sapz = z - 0.5 * (AlNzSpan + SapzSpan)
    print("Sapz:", Sapz)
    span = 10e-6 # x & y span for sapphire
    FDTDzMin = -0.5 * AlNzSpan - 0.532e-6

    FDTDzMin = -0.5 * AlNzSpan -  wv
    print("FDTDzMin:", FDTDzMin)
    FDTDzMax =  AlNzSpan * 0.5 +  wv
    print("FDTDzMax:", FDTDzMax)
    FDTDspan = 3 * wv
    print("FDTDspan:", FDTDspan)
    AlNDFTz2 = 0.5 * AlNzSpan + 0.532e-6
    print("AlNDFTz2:", AlNDFTz2)

    mesh = 0.1e-6  # only for testing - increase for final runs
    print("Mesh:", mesh)

    fdtd = lumapi.FDTD(hide = True)

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

    for i in range(n):
        fdtd.adddftmonitor()
        fdtd.set("name", f"AlNDFT{i+1}")

    configuration = (
        ("PlaneWave", (
                    ("x", x),
                    ("y", y),
                    ("z", PlaneZ),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan),
                    ("angle theta", theta),
                    ("amplitude", 3.7e8),
                    ("wavelength start", 0.73e-6),
                    ("wavelength stop", 0.73e-6))),

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
                    ("based on a structure", True),
                    ("structure", "AlNfilm"))),

        ("FDTD", (
                    ("x", x),
                    ("y", y),
                    ("x span", 1.5 * wv),
                    ("y span", FDTDspan),
                    ("z min", FDTDzMin),
                    ("z max", FDTDzMax),
                    ("x min bc", "bloch"),
                    ("y min bc", "bloch"))),

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
        fdtd.setnamed(f"AlNDFT{i+1}", "x", x)
        fdtd.setnamed(f"AlNDFT{i+1}", "y", y)
        fdtd.setnamed(f"AlNDFT{i+1}", "z", DFTz+ (-1 ** i ) * 0.005e-6 * (i+2 // 2))
        fdtd.setnamed(f"AlNDFT{i+1}", "x span", FDTDspan)
        fdtd.setnamed(f"AlNDFT{i+1}", "y span", FDTDspan)

    fdtd.save("rect-test")
    fdtd.run()

    for i in range(n):
        fdtd.eval(f"E{i+1} = rectilineardataset(\"EM Fields\", getresult(\"AlNDFT{i+1}\", \"x\"), getresult(\"AlNDFT{i+1}\", \"y\"), getresult(\"AlNDFT{i+1}\", \"z\"));")
        fdtd.eval("chi1 = 5.3458;")
        fdtd.eval(f"E{i+1}x= getresult(\"AlNDFT{i+1}\", \"Ex\");E{i+1}y= getresult(\"AlNDFT{i+1}\", \"Ey\");E{i+1}z= getresult(\"AlNDFT{i+1}\", \"Ez\");")
        fdtd.eval(f"E{i+1}x2 = (2 * {chi2} * E{i+1}z * E{i+1}x) / chi1; E{i+1}y2 = (2 * {chi2} * E{i+1}z * E{i+1}y) / chi1; E{i+1}z2 = ({chi2} * (E{i+1}x ^ 2 + E{i+1}y ^ 2) - {chi2} * E{i+1}z ^ 2) / chi1;")
        fdtd.eval(f"E{i+1}.addparameter(\"lambda\", 299792458/getresult(\"AlNDFT{i+1}\", \"f\"), \"f\", getresult(\"AlNDFT{i+1}\", \"f\"));")
        fdtd.eval(f"E{i+1}.addattribute(\"E\", E{i+1}x2, E{i+1}y2, E{i+1}z2);")

    fdtd.switchtolayout()
    fdtd.select("PlaneWave")
    fdtd.delete()

    fdtd.adddftmonitor()
    fdtd.set("name", "AlNDFTair")

    for i in range(n):
        fdtd.eval(f"addimportedsource; importdataset(E{i+1});set(\"name\", \"source{i+1}\");set(\"x\", {x});set(\"y\", {y});set(\"z\", {DFTz + (-1 ** i ) * 0.005e-6 * (i+2 // 2)});set(\"injection axis\", \"z\");set(\"direction\", \"forward\");")
        
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

    fdtd.save("rect-test2")
    fdtd.run()

    result = fdtd.getresult("AlNDFTair", "power")

    return real(result)[0][0] / 3.7e8

# 31,-5e-07,-5e-07,-2.39e-07,8.06e-07,1.46e-06,1.2e-06,6.91e-08,28.11,1.2613674309675853,success
# print(runSim1(-5e-7, -5e-7, -2.39e-7, 8.059184e-7, 1.460000e-6, 1.200321e-6, 15, 6.907945e-8, 28.106))

# MQWs:
for i in range(30):
    log_to_csv(
        -5e-07,
        -5e-07,
        -2.39e-07,
        8.059184e-07,
        1.46e-06,
        1.200321e-06,
        6.907945e-08,
        28.106,
        i + 1,
        runSim1(
            -5e-07,
            -5e-07,
            -2.39e-07,
            8.059184e-07,
            1.46e-06,
            1.200321e-06,
            i+1,
            6.907945e-08,
            28.106
        )
    )
