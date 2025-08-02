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

# x & y span up to 2 wv
def runSim1(AlNxSpan, AlNySpan, AlNzSpan, n, DFTz=0, theta=40): 

    x = 0
    y = 0
    z = 0

    if(abs(DFTz) > 0.5 * AlNzSpan):
        return 0.0

    PlaneZ = -0.5 * AlNzSpan - 0.5 * wv
    
    SapzSpan = 1 * wv  # Reduced from 2*wv
    Sapz = z - 0.5 * (AlNzSpan + SapzSpan)
    span = 2 * wv  # Reduced from 4*wv to 2*wv for sapphire
    FDTDzMin = -0.5 * AlNzSpan - 0.5 * wv

    FDTDzMin = -0.5 * AlNzSpan -  wv * 0.5
    FDTDzMax =  AlNzSpan * 0.5 +  wv * 0.5
    FDTDspan = 2 * wv  # Reduced from 3*wv to 2*wv
    AlNDFTz2 = 0.5 * AlNzSpan + 0.5 * wv
    mesh = 0.2e-6  # Increased from 0.1e-6 to 0.2e-6 for faster simulation
    
    fdtd = lumapi.FDTD(hide = True)  # Hide GUI for faster execution

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

    configuration = (
        ("PlaneWave", (
                    ("x", x),
                    ("y", y),
                    ("z", PlaneZ),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan),
                    ("angle theta", theta),
                    ("amplitude", 3.7e8),
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
        fdtd.adddftmonitor()
        fdtd.set("name", f"AlNDFT{i+1}")
        fdtd.setnamed(f"AlNDFT{i+1}", "x", x)
        fdtd.setnamed(f"AlNDFT{i+1}", "y", y)
        fdtd.setnamed(f"AlNDFT{i+1}", "z", DFTz + (-1 ** i ) * 0.005e-6 * (i+2 // 2))
        fdtd.setnamed(f"AlNDFT{i+1}", "x span", FDTDspan)
        fdtd.setnamed(f"AlNDFT{i+1}", "y span", FDTDspan)

    fdtd.save("rect-test")
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

# 37,0.345,1.46,1.355,-0.144,25.5,2.0419825923541257,success
AlNxSpan = 0.345e-6
AlNySpan = 1.46e-6
AlNZspan = 1.355e-6
DFTz = -0.144e-6
theta = 25.5

power = runSim1(AlNxSpan, AlNySpan, AlNZspan, 1, DFTz=DFTz, theta=theta)
print(f"1 QW: {power}")

power = runSim1(AlNxSpan, AlNySpan, AlNZspan, 10, DFTz=DFTz, theta=theta)
print(f"10 QW: {power}")
