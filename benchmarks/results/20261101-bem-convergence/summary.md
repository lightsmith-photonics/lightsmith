# BEM convergence: Au (JC) sphere r = 40 nm in water, wavelengths [520.0, 600.0, 700.0, 800.0] nm

Error = max over wavelengths of |C/C_Mie - 1|.

| method | variant | resolution | size | wall (s) | C_ext err | C_abs err |
|---|---|---|---|---|---|---|
| bem | default | mesh_size_nm=16 | 540 | 0.9 | 0.0564 % | 0.0552 % |
| bem | default | mesh_size_nm=12 | 960 | 2.8 | 0.0318 % | 0.0328 % |
| bem | default | mesh_size_nm=10 | 1500 | 5.1 | 0.0207 % | 0.0221 % |
| bem | default | mesh_size_nm=8 | 2160 | 4.8 | 0.0145 % | 0.0158 % |
| bem | default | mesh_size_nm=6 | 3840 | 11.0 | 0.00814 % | 0.00906 % |
| bem | default | mesh_size_nm=4 | 8640 | 49.3 | 0.00352 % | 0.00402 % |
| bem | no-volume-correction | mesh_size_nm=16 | 540 | 0.9 | 11.7 % | 7.6 % |
| bem | no-volume-correction | mesh_size_nm=12 | 960 | 2.0 | 6.79 % | 4.37 % |
| bem | no-volume-correction | mesh_size_nm=10 | 1500 | 3.6 | 4.42 % | 2.83 % |
| bem | no-volume-correction | mesh_size_nm=8 | 2160 | 4.1 | 3.09 % | 1.98 % |
| bem | no-volume-correction | mesh_size_nm=6 | 3840 | 9.6 | 1.76 % | 1.12 % |
| bem | no-volume-correction | mesh_size_nm=4 | 8640 | 83.0 | 0.785 % | 0.499 % |
| dda | center | dipoles_per_diameter=8 | 280 | 11.7 | 330 % | 1.52e+03 % |
| dda | center | dipoles_per_diameter=12 | 912 | 12.3 | 112 % | 473 % |
| dda | center | dipoles_per_diameter=16 | 2176 | 16.2 | 135 % | 508 % |
| dda | center | dipoles_per_diameter=24 | 7208 | 22.1 | 56.4 % | 209 % |
| dda | center | dipoles_per_diameter=32 | 17256 | 14.6 | 50.7 % | 199 % |
| dda | center | dipoles_per_diameter=48 | 57856 | 22.8 | 30 % | 114 % |
| dda | center | dipoles_per_diameter=64 | 137376 | 52.6 | 24.5 % | 94.8 % |
| fdtd | default | spacing_nm=8 | 166375 | 4.7 | 381 % | 1.69e+03 % |
| fdtd | default | spacing_nm=5 | 226981 | 10.2 | 206 % | 909 % |
| fdtd | default | spacing_nm=4 | 274625 | 13.4 | 309 % | 1.25e+03 % |
| fdtd | default | spacing_nm=3 | 389017 | 14.6 | 152 % | 651 % |
| fdtd | default | spacing_nm=2 | 614125 | 25.4 | 109 % | 464 % |

Cheapest run per target (headline error = max of C_ext and C_abs errors):

- 10 %: bem: 0.9 s (mesh_size_nm=16)
- 5 %: bem: 0.9 s (mesh_size_nm=16)
- 1 %: bem: 0.9 s (mesh_size_nm=16)
- 0.1 %: bem: 0.9 s (mesh_size_nm=16)
