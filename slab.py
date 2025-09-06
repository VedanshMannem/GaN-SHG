import csv
import sys, os

from numpy import delete, real, shape
from scipy import integrate
sys.path.append("C:\\Program Files\\Lumerical\\v251\\api\\python\\") 
sys.path.append(os.path.dirname(__file__)) 

from pprint import pprint
import lumapi # type: ignore
import numpy as np
import matplotlib.pyplot as plt

wv = 1.064e-6  # wavelength
chi1 = 5.3458
c = 299792458
chi2 = 1.26e-10 # quartz chi2

GaNx = 0
GaNy = 0
GaNz = 0

span = 2 * wv
GaNzSpan =  1.8e-6

PlaneZ = GaNz - 0.5 * GaNzSpan - wv # record
PlaneX = 0
PlaneY = 0

Sapx = 0
Sapy = 0
SapSpan = 2 * wv # record
Sapz = -0.5 * (GaNzSpan + SapSpan)

FDTDx = 0
FDTDy = 0
FDTDzMin = Sapz - wv
FDTDzMax = GaNz + GaNzSpan * 0.5 + wv
FDTDspan =  span # x & y span

mesh = 0.085e-6  
n=200

# Center of the GaN
GaNDFTx = 0
GaNDFTy = 0
GaNDFTz = 0

GaNDFTx2 = 0
GaNDFTy2 = 0
GaNDFTz2 = GaNz - 0.5 * GaNzSpan - wv

d = 0.18e-6  / n

def runSim1(theta, n=n):

    fdtd = lumapi.FDTD(hide = True)

    fdtd.eval("q=[2.3333;2.3333;2.3095];setmaterial(addmaterial(\"(n,k) Material\"), \"name\", \"GaN\");setmaterial(\"GaN\", \"Anisotropy\", 1);setmaterial(\"GaN\", \"Refractive Index\", q);")
    # [2.3333;2.3333;2.3095] for GaN
    # [2.1297;2.1297;2.1712] for AlN

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

    for i in range(n):
        fdtd.adddftmonitor()
        fdtd.set("name", f"GaNDFT{i+1}")

        fdtd.setnamed(f"GaNDFT{i+1}", "x", GaNDFTx)
        fdtd.setnamed(f"GaNDFT{i+1}", "y", GaNDFTy)
        fdtd.setnamed(f"GaNDFT{i+1}", "z", GaNDFTz + (-1)**i * i * d)
        fdtd.setnamed(f"GaNDFT{i+1}", "x span", FDTDspan)
        fdtd.setnamed(f"GaNDFT{i+1}", "y span", FDTDspan)

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
                    ("amplitude", 1),
                    ("wavelength start", wv),
                    ("wavelength stop", wv))),

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
                  ("x min bc", "PML"),
                  ("y min bc", "PML"))),

        ("Sapphire", ( 
                  ("x",Sapx),
                  ("y",Sapy),
                  ("z",Sapz),
                  ("x span", SapSpan),
                  ("y span", SapSpan),
                  ("z span", SapSpan),
                  ("material", "Al2O3 - Palik"))),
    )

    for obj, parameters in configuration:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)

    fdtd.save("slab-test")
    fdtd.run()

    for i in range(n):
        fdtd.eval(f"E{i+1} = rectilineardataset(\"EM Fields\", getresult(\"GaNDFT{i+1}\", \"x\"), getresult(\"GaNDFT{i+1}\", \"y\"), getresult(\"GaNDFT{i+1}\", \"z\"));")
        fdtd.eval("chi1 = 5.3458;")
        fdtd.eval(f"Ex{i+1}= getresult(\"GaNDFT{i+1}\", \"Ex\");Ey{i+1}= getresult(\"GaNDFT{i+1}\", \"Ey\");Ez{i+1}= getresult(\"GaNDFT{i+1}\", \"Ez\");")
        # fdtd.eval(f"E2x{i+1} = (2 * {chi2} * Ez{i+1} * Ex{i+1}) / chi1; E2y{i+1} = (2 * {chi2} * Ez{i+1} * Ey{i+1}) / chi1; E2z{i+1} = ({chi2} * (Ex{i+1} ^ 2 + Ey{i+1} ^ 2) - {chi2} * Ez{i+1} ^ 2) / chi1;")
        fdtd.eval(f"E2x{i+1} = (2 * Ez{i+1} * Ex{i+1}) / chi1; E2y{i+1} = (2 * Ez{i+1} * Ey{i+1}) / chi1; E2z{i+1} = ((Ex{i+1} ^ 2 + Ey{i+1} ^ 2) - Ez{i+1} ^ 2) / chi1;")
        fdtd.eval(f"E{i+1}.addparameter(\"lambda\", 299792458/getresult(\"GaNDFT{i+1}\", \"f\"), \"f\", getresult(\"GaNDFT{i+1}\", \"f\"));")
        fdtd.eval(f"E{i+1}.addattribute(\"E\", E2x{i+1}, E2y{i+1}, E2z{i+1});")
        fdtd.eval(f"E{i+1}.addattribute(\"H\", 0, 0, 0);")

    fdtd.switchtolayout()
    fdtd.select("PlaneWave")
    fdtd.delete()

    fdtd.adddftmonitor()
    fdtd.set("name", "GaNDFTfinal")

    for i in range(n):
        fdtd.eval(f"addimportedsource; importdataset(E{i+1});set(\"name\", \"source{i+1}\");set(\"x\", {GaNDFTx});set(\"y\", {GaNDFTy});set(\"z\", {GaNDFTz + (-1)**i * i * d});set(\"injection axis\", \"z\");set(\"direction\", \"forward\");")

    configuration2 = (
        ("GaNDFTfinal", (
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

    result = fdtd.getresult("GaNDFTfinal", "power")
    print(result)

    return real(result)[0][0] 

# power = runSim1(theta=30)
# print(f"Parameters used: \n span: {span} \n GaNzSpan: {GaNzSpan} \n mesh: {mesh} \n Sapphire: {SapSpan} \n everything else: wv")
# print("Power at 2nd monitor:", power, " with n =", n)

# Plotting
for i in range(39, 91):
    power = runSim1(i)
    with open('SHG.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([i, power])
    print(f"Angle: {i}, Power: {power}\n")
