# Benchmark summary

98 runs (98 ok). Headline error = max relative error over wavelengths of C_ext and C_abs against the reference.

| problem | reference | backend / variant / device | finest res | error | wall s |
|---|---|---|---|---|---|
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-dda / center / gpu | 24 | 0.678 | 24.00 |
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-fdtd / default / gpu | 4 | 3 | 7.63 |
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-mie / default / gpu | nan | 5.64e-15 | 0.78 |
| coreshell-sio2-au-c30-s8-water | Mie (omniscatter-mie) | omniscatter-quasistatic / default / cpu | nan | 1.55 | 0.00 |
| dimer-au-r15-gap4-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / center / cpu | 8 | 0.636 | 11.49 |
| dimer-au-r15-gap4-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / center / gpu | 16 | 0.303 | 2.97 |
| dimer-au-r15-gap4-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-fdtd / default / gpu | 4 | 0.382 | 7.30 |
| rod-au-ar3-r8-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=20, gpu | omniscatter-dda / center / cpu | 16 | 2.53 | 7.01 |
| rod-au-ar3-r8-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=20, gpu | omniscatter-dda / center / gpu | 16 | 2.53 | 3.88 |
| rod-au-ar3-r8-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=20, gpu | omniscatter-fdtd / default / gpu | 3.2 | 0.491 | 13.60 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-dda / center / cpu | 8 | 55.5 | 13.55 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-dda / center / gpu | 24 | 2.92 | 22.17 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-fdtd / default / gpu | 12 | 4.21 | 7.31 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-mie / default / cpu | nan | 1.82e-12 | 0.01 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-mie / default / gpu | nan | 1.82e-12 | 0.07 |
| sphere-ag-jc-r30-vacuum | Mie (miepython) | omniscatter-quasistatic / default / cpu | nan | 1.63 | 0.00 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-dda / center / cpu | 8 | 4.18 | 3.25 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-dda / center / gpu | 24 | 1.32 | 2.22 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-fdtd / default / gpu | 8 | 11.8 | 3.32 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-mie / default / cpu | nan | 6.05e-12 | 0.01 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-mie / default / gpu | nan | 6.05e-12 | 0.02 |
| sphere-au-jc-r20-water | Mie (miepython) | omniscatter-quasistatic / default / cpu | nan | 0.157 | 0.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-dda / center / cpu | 8 | 0.0475 | 0.15 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-dda / center / gpu | 20 | 0.0143 | 0.39 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-dda / weighted / cpu | 8 | 0.0491 | 0.51 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-dda / weighted / gpu | 20 | 0.0222 | 0.93 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-fdtd / default / cpu | 10.77 | 0.283 | 39.83 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-fdtd / default / gpu | 10.77 | 0.283 | 3.18 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-mie / default / cpu | nan | 3.03e-12 | 0.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-mie / default / gpu | nan | 3.03e-12 | 0.05 |
| sphere-diel-x1.5-m1.5-k0.01-water | Mie (miepython) | omniscatter-quasistatic / default / cpu | nan | 1.06 | 0.00 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / center / cpu | 10 | 12.1 | 14.16 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / center / gpu | 32 | 1.57 | 5.09 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / extrapolate / gpu | 32 | 9.75 | 10.51 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / weighted / cpu | 8 | 1.78 | 47.50 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-dda / weighted / gpu | 32 | 2.24 | 8.37 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-fdtd / default / gpu | 2 | 2.41 | 13.94 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-mie / default / cpu | nan | 1.17e-12 | 0.01 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-mie / default / gpu | nan | 1.17e-12 | 0.02 |
| sphere-drude-r40-water | Mie (miepython) | omniscatter-quasistatic / default / cpu | nan | 0.408 | 0.00 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / center / cpu | 16 | 0.466 | 12.98 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-dda / center / gpu | 16 | 0.466 | 3.10 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-fdtd / default / gpu | 6.35 | 4.19 | 5.80 |
| spheroid-au-ar2-req20-water | highest-resolution omniscatter-dda (center) @ dipoles_per_diameter=24, gpu | omniscatter-quasistatic / default / cpu | nan | 0.806 | 0.00 |

## Convergence (headline error)

| problem | backend / variant / device | resolutions | order p | noise (dex) | monotone frac |
|---|---|---|---|---|---|
| coreshell-sio2-au-c30-s8-water | omniscatter-dda / center / gpu | 12 16 24 | 0.96 | 0.52 | 0.50 |
| dimer-au-r15-gap4-water | omniscatter-dda / center / gpu | 8 12 16 | 1.43 | 0.78 | 0.50 |
| rod-au-ar3-r8-water | omniscatter-dda / center / cpu | 8 12 16 | -2.18 | 0.18 | 0.00 |
| rod-au-ar3-r8-water | omniscatter-dda / center / gpu | 8 12 16 | -2.18 | 0.18 | 0.00 |
| rod-au-ar3-r8-water | omniscatter-fdtd / default / gpu | 4 3.2 | nan | nan | 0.00 |
| sphere-ag-jc-r30-vacuum | omniscatter-dda / center / gpu | 8 12 16 24 | 2.34 | 0.50 | 0.67 |
| sphere-au-jc-r20-water | omniscatter-dda / center / gpu | 8 12 16 24 | 0.80 | 0.26 | 0.67 |
| sphere-diel-x1.5-m1.5-k0.01-water | omniscatter-dda / center / gpu | 8 12 16 20 | 1.30 | 0.01 | 1.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | omniscatter-dda / weighted / gpu | 8 12 16 20 | 0.88 | 0.04 | 1.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | omniscatter-fdtd / default / cpu | 21.54 10.77 | nan | nan | 1.00 |
| sphere-diel-x1.5-m1.5-k0.01-water | omniscatter-fdtd / default / gpu | 21.54 10.77 | nan | nan | 1.00 |
| sphere-drude-r40-water | omniscatter-dda / center / cpu | 8 10 | nan | nan | 1.00 |
| sphere-drude-r40-water | omniscatter-dda / center / gpu | 8 10 12 14 16 18 20 22 24 28 32 | 1.46 | 0.26 | 0.60 |
| sphere-drude-r40-water | omniscatter-dda / extrapolate / gpu | 16 24 32 | 3.07 | 0.78 | 0.50 |
| sphere-drude-r40-water | omniscatter-dda / weighted / gpu | 8 12 16 32 | -0.11 | 0.36 | 0.33 |
| sphere-drude-r40-water | omniscatter-fdtd / default / gpu | 8 4 2 | 1.04 | 0.10 | 1.00 |
| spheroid-au-ar2-req20-water | omniscatter-dda / center / cpu | 8 12 16 | 0.28 | 0.08 | 0.50 |
| spheroid-au-ar2-req20-water | omniscatter-dda / center / gpu | 8 12 16 | 0.28 | 0.08 | 0.50 |
