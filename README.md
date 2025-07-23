# Optimizing SHG in Gallium Nitride
Repo contains all the files for optimizing the Second Harmonic Generation of Gallium Nitride with circular, triangular, and rectangular metasurfaces. ".fsp" sample files are also included to run simulations in Lumerical. Here are some sample images of the simulations:

<img width="1554" height="783" alt="image" src="https://github.com/user-attachments/assets/c986d9fc-733f-4924-86c5-8a291e34c292" />
<img width="1470" height="789" alt="Screenshot 2025-07-19 214333" src="https://github.com/user-attachments/assets/ab86e64e-cb4b-45d1-a955-19224c845484" />

The goal of this project is to otpimize Second Harmonic Generation (SHG). SHG is a process by which an input source of frequency omega gets doubled when it interacts with an anisotropic material (asymmetric about the origin). GaN is popular becuase its a dielectric material with a higher index of refraction, making it ideal to perform SHG. The project uses Bayesian Optimization and basic Reinforcement learning to learn and train different parameters. SHG efficiencies reach a peak of around 8.8 x 10^-9 at a wavelength of 1064 nm, showcasing the viability of GaN for nonlinear optical processes such as Spontaneous Paramteric Down Conversion on quantum chips.

## To Use
**Requirement**: Need a Lumerical license to use the software on your machine

Tweak the refraction indices based on metals being used

Code utilizes the lumapi package which is available when Lumerical is installed

## .csv files
Contains data from previous training runs

