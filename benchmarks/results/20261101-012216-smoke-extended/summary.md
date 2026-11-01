# Benchmark summary

154 runs (154 ok). Headline error = max relative error over wavelengths of C_ext and C_abs against the reference.

| problem | reference | backend / variant / device | finest res | error | wall s |
|---|---|---|---|---|---|
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-dda / center / cpu | 12 | 1.12 | 158.72 |
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-dda / center / gpu | 32 | 1.27 | 31.15 |
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-dda / weighted / cpu | 8 | 0.898 | 13.97 |
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-dda / weighted / gpu | 32 | nan | 93.68 |
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-fdtd / default / gpu | 2 | 3.23 | 64.61 |
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-mie / default / gpu | nan | 5.64e-15 | 1.05 |
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-quasistatic / default / cpu | nan | 1.55 | 0.00 |
| dimer-au-r15-gap4-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / center / cpu | 8 | 0.636 | 14.23 |
| dimer-au-r15-gap4-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / center / gpu | 16 | 0.303 | 3.13 |
| dimer-au-r15-gap4-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / weighted / cpu | 8 | 1.29 | 91.51 |
| dimer-au-r15-gap4-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / weighted / gpu | 24 | nan | 67.80 |
| dimer-au-r15-gap4-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-fdtd / default / gpu | 4 | 0.382 | 6.87 |
| rod-au-ar3-r8-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=20, cpu | omniscatter-dda / center / cpu | 16 | 2.53 | 10.16 |
| rod-au-ar3-r8-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=20, cpu | omniscatter-dda / center / gpu | 20 | 4.22e-06 | 6.22 |
| rod-au-ar3-r8-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=20, cpu | omniscatter-dda / weighted / cpu | 20 | 0.905 | 85.96 |
| rod-au-ar3-r8-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=20, cpu | omniscatter-dda / weighted / gpu | 20 | 0.905 | 19.81 |
| rod-au-ar3-r8-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=20, cpu | omniscatter-fdtd / default / gpu | 3.2 | 0.491 | 20.45 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-dda / center / cpu | 12 | 2.47 | 46.91 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-dda / center / gpu | 24 | 2.92 | 8.35 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-dda / weighted / cpu | 8 | 12.4 | 26.19 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-dda / weighted / gpu | 24 | 4.05 | 46.30 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-fdtd / default / gpu | 6 | 18.6 | 6.78 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-mie / default / cpu | nan | 1.82e-12 | 0.00 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-mie / default / gpu | nan | 1.82e-12 | 0.02 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-quasistatic / default / cpu | nan | 1.63 | 0.00 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-dda / center / cpu | 12 | 1.1 | 10.53 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-dda / center / gpu | 24 | 1.32 | 2.51 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-dda / weighted / cpu | 8 | 1.94 | 8.74 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-dda / weighted / gpu | 24 | 1.32 | 12.27 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-fdtd / default / gpu | 4 | 1.81 | 10.51 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-mie / default / cpu | nan | 6.05e-12 | 0.01 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-mie / default / gpu | nan | 6.05e-12 | 0.02 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-quasistatic / default / cpu | nan | 0.157 | 0.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-dda / center / cpu | 12 | 0.0273 | 0.48 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-dda / center / gpu | 20 | 0.0143 | 0.67 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-dda / weighted / cpu | 8 | 0.0491 | 0.46 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-dda / weighted / gpu | 20 | 0.0222 | 1.31 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-fdtd / default / cpu | 10.77 | 0.283 | 50.28 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-fdtd / default / gpu | 10.77 | 0.283 | 6.38 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-mie / default / cpu | nan | 3.03e-12 | 0.04 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-mie / default / gpu | nan | 3.03e-12 | 0.03 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-quasistatic / default / cpu | nan | 1.06 | 0.00 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / center / cpu | 12 | 1.3 | 21.49 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / center / gpu | 32 | 1.57 | 3.26 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / extrapolate / gpu | 32 | 9.75 | 16.79 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / weighted / cpu | 10 | 4.23 | 59.80 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / weighted / gpu | 32 | 2.24 | 18.03 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-fdtd / default / gpu | 2 | 2.41 | 42.49 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-mie / default / cpu | nan | 1.17e-12 | 0.02 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-mie / default / gpu | nan | 1.17e-12 | 0.09 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-quasistatic / default / cpu | nan | 0.408 | 0.02 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / center / cpu | 16 | 0.466 | 11.08 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / center / gpu | 16 | 0.466 | 8.25 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / weighted / cpu | 16 | 1.54 | 67.05 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / weighted / gpu | 24 | 0.967 | 15.24 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-fdtd / default / gpu | 3.175 | 0.329 | 15.65 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-quasistatic / default / cpu | nan | 0.806 | 0.00 |

## Convergence (headline error)

| problem | backend / variant / device | resolutions | order p | noise (dex) | monotone frac |
|---|---|---|---|---|---|
| coreshell-sio2-au-c30-s8-water | omniscatter-dda / center / cpu | 8 12 | nan | nan | 1.00 |
| coreshell-sio2-au-c30-s8-water | omniscatter-dda / center / gpu | 12 16 24 32 | 0.52 | 0.38 | 0.33 |
| coreshell-sio2-au-c30-s8-water | omniscatter-fdtd / default / gpu | 4 2 | nan | nan | 0.00 |
| dimer-au-r15-gap4-water | omniscatter-dda / center / gpu | 8 12 16 | 1.43 | 0.78 | 0.50 |
| dimer-au-r15-gap4-water | omniscatter-dda / weighted / gpu | 8 12 16 | 3.70 | 0.46 | 1.00 |
| rod-au-ar3-r8-water | omniscatter-dda / center / cpu | 8 12 16 | -2.18 | 0.18 | 0.00 |
| rod-au-ar3-r8-water | omniscatter-dda / center / gpu | 8 12 16 20 | 10.12 | 2.68 | 0.33 |
| rod-au-ar3-r8-water | omniscatter-dda / weighted / cpu | 8 12 16 20 | -0.57 | 0.39 | 0.33 |
| rod-au-ar3-r8-water | omniscatter-dda / weighted / gpu | 8 12 16 20 | -0.57 | 0.39 | 0.33 |
| rod-au-ar3-r8-water | omniscatter-fdtd / default / gpu | 4 3.2 | nan | nan | 0.00 |
| sphere-ag-jc-r30-vacuum | omniscatter-dda / center / cpu | 8 12 | nan | nan | 1.00 |
| sphere-ag-jc-r30-vacuum | omniscatter-dda / center / gpu | 8 12 16 24 | 2.34 | 0.50 | 0.67 |
| sphere-ag-jc-r30-vacuum | omniscatter-dda / weighted / gpu | 8 24 | nan | nan | 1.00 |
| sphere-ag-jc-r30-vacuum | omniscatter-fdtd / default / gpu | 12 6 | nan | nan | 0.00 |
| sphere-au-jc-r20-water | omniscatter-dda / center / cpu | 8 12 | nan | nan | 1.00 |
| sphere-au-jc-r20-water | omniscatter-dda / center / gpu | 8 12 16 24 | 0.80 | 0.26 | 0.67 |
| sphere-au-jc-r20-water | omniscatter-dda / weighted / gpu | 8 12 16 24 | 0.55 | 0.27 | 0.67 |
| sphere-au-jc-r20-water | omniscatter-fdtd / default / gpu | 8 4 | nan | nan | 1.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | omniscatter-dda / center / cpu | 8 12 | nan | nan | 1.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | omniscatter-dda / center / gpu | 8 12 16 20 | 1.30 | 0.01 | 1.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | omniscatter-dda / weighted / gpu | 8 12 16 20 | 0.88 | 0.04 | 1.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | omniscatter-fdtd / default / cpu | 21.54 10.77 | nan | nan | 1.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | omniscatter-fdtd / default / gpu | 21.54 10.77 | nan | nan | 1.00 |
| sphere-drude-r40-water | omniscatter-dda / center / cpu | 8 10 12 | 5.81 | 0.40 | 1.00 |
| sphere-drude-r40-water | omniscatter-dda / center / gpu | 8 10 12 14 16 18 20 22 24 28 32 | 1.46 | 0.26 | 0.60 |
| sphere-drude-r40-water | omniscatter-dda / extrapolate / gpu | 16 24 32 | 3.07 | 0.78 | 0.50 |
| sphere-drude-r40-water | omniscatter-dda / weighted / cpu | 8 10 | nan | nan | 0.00 |
| sphere-drude-r40-water | omniscatter-dda / weighted / gpu | 8 10 12 14 16 18 20 28 32 | 0.30 | 0.24 | 0.38 |
| sphere-drude-r40-water | omniscatter-fdtd / default / gpu | 8 4 2 | 1.04 | 0.10 | 1.00 |
| spheroid-au-ar2-req20-water | omniscatter-dda / center / cpu | 8 12 16 | 0.28 | 0.08 | 0.50 |
| spheroid-au-ar2-req20-water | omniscatter-dda / center / gpu | 8 12 16 | 0.28 | 0.08 | 0.50 |
| spheroid-au-ar2-req20-water | omniscatter-dda / weighted / cpu | 8 12 16 | 2.96 | 0.60 | 0.50 |
| spheroid-au-ar2-req20-water | omniscatter-dda / weighted / gpu | 12 16 24 | -0.56 | 0.26 | 0.50 |
| spheroid-au-ar2-req20-water | omniscatter-fdtd / default / gpu | 6.35 3.175 | nan | nan | 1.00 |
