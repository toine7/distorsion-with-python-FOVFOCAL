"""
FOV calibration (FOVCAL-style) from an image of a dot grid target.

The nominal grid is fitted to the measured dot positions with a similarity
(translation, rotation, scale) or affine transform. The fit scale gives the
camera scale and, with the sensor pixel pitch, the magnification. The fit
residuals contain the nonlinear lens aberrations, including distortion.

Python port of the MATLAB script (imfindcircles + fitgeotrans) with:
  - sub-pixel dot centres (intensity-weighted centroids)
  - robust grid indexing (works even if the grid is rotated or a dot is missing)
  - similarity/affine fit -> magnification, residual vectors = distortion map
  - radial distortion fit (k1, k2) -> distortion in % at the edge of the field
  - results saved as CSV + PNG

Requires: numpy, scipy, matplotlib, pillow
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage, optimize, interpolate

# %% Parameters
grid_img_path = 'Grid.tif'
radiusRange_pixs = (24, 85)     # accepted dot radius range [pix]
dotSpacing_mm = 0.8             # grid pitch on the target [mm]
pixelSize_mm = 5.5e-3           # camera pixel size [mm]
# 'similarity': translation + rotation + one scale (4 params, FOVCAL definition)
# 'affine'    : + separate x/y scale and shear (6 params, like MATLAB fitgeotrans 'affine')
fit_type = 'similarity'
out_dir = 'results'

os.makedirs(out_dir, exist_ok=True)

# %% Load image
I = np.array(Image.open(grid_img_path).convert('L'), dtype=float)
resY, resX = I.shape

# %% Detect dots (bright blobs)
# Otsu threshold, then label connected regions
hist, edges = np.histogram(I, bins=256)
mids = 0.5 * (edges[:-1] + edges[1:])
w0 = np.cumsum(hist)
w1 = w0[-1] - w0
m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
thr = mids[np.argmax(w0 * w1 * (m0 - m1) ** 2)]

mask = ndimage.binary_fill_holes(I > thr)   # fill dust specks inside the dots
labels, nLab = ndimage.label(mask)
idx = np.arange(1, nLab + 1)

# intensity-weighted centroid (background removed) -> sub-pixel centres
bg = np.median(I[~mask])
weights = np.clip(I - bg, 0, None)
cy_cx = np.array(ndimage.center_of_mass(weights, labels, idx))   # (row, col)
centers = cy_cx[:, ::-1]                                          # (x, y) like MATLAB
areas = ndimage.sum(np.ones_like(I), labels, idx)
radii = np.sqrt(areas / np.pi)

# %% Filter dots
# 1) radius in range, 2) not cut by the border (bounding box touches the edge)
slices = ndimage.find_objects(labels)
touches_border = np.array([s[0].start == 0 or s[1].start == 0 or
                           s[0].stop == resY or s[1].stop == resX for s in slices])
keep = (radii >= radiusRange_pixs[0]) & (radii <= radiusRange_pixs[1]) & ~touches_border
# 3) also keep a margin of one mean radius from the border (as in the MATLAB code)
meanRadius = radii[keep].mean()
margin = meanRadius
keep &= ((centers[:, 0] > margin) & (centers[:, 0] < resX - margin) &
         (centers[:, 1] > margin) & (centers[:, 1] < resY - margin))
# 4) reject size outliers (merged / partial blobs)
medR = np.median(radii[keep])
keep &= np.abs(radii - medR) < 0.2 * medR

centers_filt = centers[keep]
radii_filt = radii[keep]
print(f'Dots detected: {nLab}, kept after filtering: {len(centers_filt)}')

# %% Show filtered centres
fig1, ax = plt.subplots(figsize=(8, 8))
ax.imshow(I, cmap='gray')
for (x, y), r in zip(centers_filt, radii_filt):
    ax.add_patch(plt.Circle((x, y), r, fill=False, color='g', lw=1))
ax.plot(centers_filt[:, 0], centers_filt[:, 1], 'r+', ms=8, mew=1.5)
ax.set_title('Filtered circle centers')
fig1.savefig(os.path.join(out_dir, '1_detected_dots.png'), dpi=150, bbox_inches='tight')

# %% Assign grid indices (i = column, j = row)
# Estimate the two lattice vectors from nearest-neighbour displacements.
# This replaces the "sort into rows" step, which breaks when the grid is
# slightly rotated or when a dot is missing at the start of a row.
pts = centers_filt
d = pts[None, :, :] - pts[:, None, :]
dist = np.hypot(d[..., 0], d[..., 1])
np.fill_diagonal(dist, np.inf)
pitch_pix = np.median(dist.min(axis=1))

nn = d[(dist > 0.7 * pitch_pix) & (dist < 1.3 * pitch_pix)]      # all 1st-neighbour vectors
ang = np.arctan2(nn[:, 1], nn[:, 0])
# vector along x (|angle| < 45 deg) and along y (45..135 deg)
vx = nn[np.abs(ang) < np.pi / 4]
vy = nn[(ang > np.pi / 4) & (ang < 3 * np.pi / 4)]
ex = np.median(vx, axis=0)
ey = np.median(vy, axis=0)
B = np.column_stack([ex, ey])

# origin = dot closest to the image centre, then integer coordinates
c0 = pts[np.argmin(np.hypot(pts[:, 0] - resX / 2, pts[:, 1] - resY / 2))]
ij = np.linalg.solve(B, (pts - c0).T).T
ij_round = np.round(ij)
if np.max(np.abs(ij - ij_round)) > 0.25:
    print('WARNING: some dots are far from the ideal lattice, check figure 1')
ij_round -= ij_round.min(axis=0)       # start indices at 0 (top-left dot)

nominalPoints = ij_round * dotSpacing_mm      # [mm]  (x right, y down, like the image)
measuredPoints = pts                          # [pix]
print(f'Grid: {int(ij_round[:, 0].max()) + 1} columns x {int(ij_round[:, 1].max()) + 1} rows')

# %% Fit transform  (nominal [mm] -> measured [pix])
# measured = [x_mm, y_mm, 1] @ T   (same convention as MATLAB fitgeotrans: tform.T)
X = np.column_stack([nominalPoints, np.ones(len(nominalPoints))])
if fit_type == 'affine':
    T, *_ = np.linalg.lstsq(X, measuredPoints, rcond=None)     # 3x2
elif fit_type == 'similarity':
    # x' = a x - b y + tx,  y' = b x + a y + ty   with a = s cos(th), b = s sin(th)
    nx, ny = nominalPoints[:, 0], nominalPoints[:, 1]
    one, zero = np.ones_like(nx), np.zeros_like(nx)
    M = np.vstack([np.column_stack([nx, -ny, one, zero]),
                   np.column_stack([ny, nx, zero, one])])
    (a, b, tx, ty), *_ = np.linalg.lstsq(M, measuredPoints.T.ravel(), rcond=None)
    T = np.array([[a, b], [-b, a], [tx, ty]])
else:
    raise ValueError(f'unknown fit_type: {fit_type}')
T = np.column_stack([T, [0, 0, 1]])                        # 3x3 like MATLAB

# %% Compute fit and residuals
measFit = X @ T[:, :2]
residuals = measuredPoints - measFit                       # [pix]
res_norm = np.hypot(residuals[:, 0], residuals[:, 1])

# %% Magnification
A = T[:2, :2]
scale_pix_per_mm = np.sqrt(abs(np.linalg.det(A)))
magnification = scale_pix_per_mm * pixelSize_mm
sx = np.hypot(*A[0])    # scale along grid x [pix/mm]
sy = np.hypot(*A[1])    # scale along grid y [pix/mm]
rot_deg = np.degrees(np.arctan2(A[0, 1], A[0, 0]))

# %% Radial distortion fit (Brown model, distortion centre free)
# measured = c + (p - c) * (1 + k1 r^2 + k2 r^4),  p = linear-mapped nominal point,
# r normalised by the image half-diagonal so k1, k2 are dimensionless.
# The linear part p is the scale at the distortion centre (paraxial scale): it does
# not depend on which part of the field is covered by dots.
rNorm = np.hypot(resX, resY) / 2


def radial_model(params, P):
    cx, cy, k1, k2 = params[:4]
    T6 = params[4:].reshape(3, 2)
    p = np.column_stack([P, np.ones(len(P))]) @ T6
    q = (p - [cx, cy]) / rNorm
    r2 = np.sum(q ** 2, axis=1, keepdims=True)
    return [cx, cy] + q * rNorm * (1 + k1 * r2 + k2 * r2 ** 2)


p0 = np.concatenate([[resX / 2, resY / 2, 0, 0], T[:, :2].ravel()])
sol = optimize.least_squares(lambda p: (radial_model(p, nominalPoints) - measuredPoints).ravel(), p0)
cx, cy, k1, k2 = sol.x[:4]
T6 = sol.x[4:].reshape(3, 2)
res_rad = measuredPoints - radial_model(sol.x, nominalPoints)
res_rad_norm = np.hypot(res_rad[:, 0], res_rad[:, 1])

scale_par = np.sqrt(abs(np.linalg.det(T6[:2])))
mag_par = scale_par * pixelSize_mm

# classical distortion of each dot [%] = (real radius - paraxial radius) / paraxial radius
p_lin = X @ T6
r_lin = np.hypot(p_lin[:, 0] - cx, p_lin[:, 1] - cy)
r_meas = np.hypot(measuredPoints[:, 0] - cx, measuredPoints[:, 1] - cy)
dist_pct_meas = 100 * (r_meas - r_lin) / r_lin

# distortion in % at the farthest dot and at the image corner (model)
r_dots = r_meas.max() / rNorm
dist_pct_dots = 100 * (k1 * r_dots**2 + k2 * r_dots**4)
corner_r = max(np.hypot(x - cx, y - cy) for x in (0, resX) for y in (0, resY)) / rNorm
dist_pct_corner = 100 * (k1 * corner_r**2 + k2 * corner_r**4)

rms = lambda v: np.sqrt(np.mean(v ** 2))
um_per_pix = 1e3 / scale_pix_per_mm

# %% Summary (console, summary.txt and first page of the PDF report)
summary = f"""FOV CALIBRATION REPORT
{'=' * 60}
Image                       : {grid_img_path}  ({resX} x {resY} pix)
Date                        : {datetime.now():%Y-%m-%d %H:%M}
Dot spacing (target)        : {dotSpacing_mm} mm
Sensor pixel size           : {pixelSize_mm * 1e3:.2f} um
Dots used                   : {len(measuredPoints)}  ({int(ij_round[:, 0].max()) + 1} columns x {int(ij_round[:, 1].max()) + 1} rows)

1) FOVCAL FIT  (nominal grid -> measured grid, {fit_type})
{'-' * 60}
Transform matrix (MATLAB tform.T convention):
{np.array2string(T, precision=5, suppress_small=True)}
Scale                       : {scale_pix_per_mm:.3f} pix/mm   (x {sx:.3f}, y {sy:.3f})
MAGNIFICATION               : {magnification:.5f}   (x {sx * pixelSize_mm:.5f}, y {sy * pixelSize_mm:.5f})
Pixel size on object        : {um_per_pix:.4f} um/pix
Field of view on object     : {resX * um_per_pix / 1e3:.3f} x {resY * um_per_pix / 1e3:.3f} mm
Grid rotation               : {rot_deg:.4f} deg
Residuals (= lens aberrations, incl. distortion)
   RMS                      : {rms(res_norm):.3f} pix  = {rms(res_norm) * um_per_pix:.2f} um on object
   max                      : {res_norm.max():.3f} pix  = {res_norm.max() * um_per_pix:.2f} um on object

2) RADIAL DISTORTION MODEL  (measured = c + (p - c)(1 + k1 r^2 + k2 r^4))
{'-' * 60}
Distortion centre (x, y)    : ({cx:.1f}, {cy:.1f}) pix
   offset from image centre : ({cx - resX / 2:+.1f}, {cy - resY / 2:+.1f}) pix
k1, k2  (r norm. {rNorm:.1f} pix): {k1:.4e}, {k2:.4e}
Paraxial magnification      : {mag_par:.5f}   (scale at the distortion centre)
Distortion at farthest dot  : {dist_pct_dots:+.4f} %   ({'barrel' if dist_pct_dots < 0 else 'pincushion'})
Distortion at image corner  : {dist_pct_corner:+.4f} %   (extrapolated)
Residuals after radial model: RMS {rms(res_rad_norm):.3f} pix, max {res_rad_norm.max():.3f} pix
   (what is left: non-radial errors such as target tilt, target errors, noise)

HOW TO READ THE VECTOR MAPS
{'-' * 60}
- FOVCAL residual map: measured - fitted position. The fit takes the
  average scale over the dots, so with pincushion distortion the vectors
  point inward in the middle and outward in the corners (barrel: reversed).
- Classical distortion map: position relative to the paraxial (centre)
  scale. Pure radial distortion gives vectors pointing away from the
  centre (pincushion) or towards it (barrel), growing with the radius.
- Arrows are magnified; the key above each map gives the arrow length.
"""
print('\n' + summary)
with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
    f.write(summary)

# %% Report figures
figs = [fig1]


def vector_map(ax, P, V, title, cmap='viridis'):
    """Vector field of V at points P, over an interpolated |V| colour map."""
    mag = np.hypot(V[:, 0], V[:, 1])
    gx, gy = np.meshgrid(np.linspace(0, resX, 300), np.linspace(0, resY, 300))
    # interpolate the components, not |V|, so the colour map stays >= 0
    gvx = interpolate.griddata(P, V[:, 0], (gx, gy), method='cubic')
    gvy = interpolate.griddata(P, V[:, 1], (gx, gy), method='cubic')
    cf = ax.contourf(gx, gy, np.hypot(gvx, gvy), levels=np.linspace(0, mag.max(), 21), cmap=cmap)
    s = 0.8 * pitch_pix / max(mag.max(), 1e-9)               # arrow magnification
    q = ax.quiver(P[:, 0], P[:, 1], V[:, 0] * s, V[:, 1] * s, angles='xy',
                  scale_units='xy', scale=1, color='k', width=0.003)
    key = float(f'{mag.max() / 2:.1g}')
    ax.quiverkey(q, 0.80, 1.025, key * s, f'{key:g} pix ({key * um_per_pix:.2g} um)',
                 labelpos='E', coordinates='axes')
    ax.set_xlim(0, resX)
    ax.set_ylim(resY, 0)
    ax.set_aspect('equal')
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    ax.set_title(title, loc='left', fontsize=10)
    return cf


def component_maps(fig, P, V, what):
    lim = np.abs(V).max()
    gx, gy = np.meshgrid(np.linspace(0, resX, 300), np.linspace(0, resY, 300))
    for k, (comp, name) in enumerate(zip(V.T, ('x', 'y'))):
        ax = fig.add_subplot(1, 2, k + 1)
        g = interpolate.griddata(P, comp, (gx, gy), method='cubic')
        cf = ax.contourf(gx, gy, g, levels=np.linspace(-lim, lim, 21), cmap='RdBu_r')
        ax.plot(P[:, 0], P[:, 1], 'k.', ms=1.5)
        ax.set_xlim(0, resX)
        ax.set_ylim(resY, 0)
        ax.set_aspect('equal')
        ax.set_title(f'{what} - {name} component')
        ax.set_xlabel('x [pix]')
        fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.12, label=f'd{name} [pix]',
                     format='%.2f', ticks=np.linspace(-lim, lim, 5))


# FOVCAL residual vector field
fig, ax = plt.subplots(figsize=(9, 8))
cf = vector_map(ax, measuredPoints, residuals,
                f'FOVCAL residuals ({fit_type} fit) = lens aberrations\n'
                f'RMS {rms(res_norm):.3f} pix, max {res_norm.max():.3f} pix')
fig.colorbar(cf, ax=ax, label='|residual| [pix]', format='%.2f')
fig.savefig(os.path.join(out_dir, '2_residual_vectors.png'), dpi=150, bbox_inches='tight')
figs.append(fig)

# x / y components of the residuals
fig = plt.figure(figsize=(11, 6.5))
component_maps(fig, measuredPoints, residuals, 'FOVCAL residual')
fig.savefig(os.path.join(out_dir, '3_residual_components.png'), dpi=150, bbox_inches='tight')
figs.append(fig)

# classical distortion field over the full sensor (radial model)
fig, ax = plt.subplots(figsize=(9, 8))
gx, gy = np.meshgrid(np.linspace(0, resX, 300), np.linspace(0, resY, 300))
gr2 = ((gx - cx) ** 2 + (gy - cy) ** 2) / rNorm ** 2
cf = ax.contourf(gx, gy, 100 * (k1 * gr2 + k2 * gr2 ** 2), levels=20, cmap='viridis')
cl = ax.contour(gx, gy, 100 * (k1 * gr2 + k2 * gr2 ** 2), levels=8, colors='w', linewidths=0.6)
ax.clabel(cl, fmt='%.2f %%', fontsize=7)
vx, vy = np.meshgrid(np.linspace(64, resX - 64, 15), np.linspace(64, resY - 64, 15))
vr2 = ((vx - cx) ** 2 + (vy - cy) ** 2) / rNorm ** 2
dx, dy = (vx - cx) * (k1 * vr2 + k2 * vr2 ** 2), (vy - cy) * (k1 * vr2 + k2 * vr2 ** 2)
dmax = np.hypot(dx, dy).max()
s = 0.8 * (resX / 14) / dmax
q = ax.quiver(vx, vy, dx * s, dy * s, angles='xy', scale_units='xy', scale=1, color='k', width=0.003)
key = float(f'{dmax / 2:.1g}')
ax.quiverkey(q, 0.80, 1.025, key * s, f'{key:g} pix', labelpos='E', coordinates='axes')
x0, x1 = measuredPoints[:, 0].min(), measuredPoints[:, 0].max()
y0, y1 = measuredPoints[:, 1].min(), measuredPoints[:, 1].max()
ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'w--', lw=1, label='area covered by dots')
ax.plot(cx, cy, 'r+', ms=14, mew=2, label='distortion centre')
ax.plot(resX / 2, resY / 2, 'wx', ms=8, mew=1.5, label='image centre')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8,
          facecolor='0.45', labelcolor='w')
ax.set_xlim(0, resX)
ax.set_ylim(resY, 0)
ax.set_aspect('equal')
ax.set_xlabel('x [pix]')
ax.set_ylabel('y [pix]')
ax.set_title(f'Classical distortion map (radial model, relative to paraxial scale)\n'
             f'{dist_pct_dots:+.3f} % at farthest dot, {dist_pct_corner:+.3f} % at image corner',
             loc='left', fontsize=10)
fig.colorbar(cf, ax=ax, label='distortion [%]')
fig.savefig(os.path.join(out_dir, '4_distortion_map.png'), dpi=150, bbox_inches='tight')
figs.append(fig)

# distortion curve
fig, ax = plt.subplots(figsize=(9, 5.5))
ok = r_lin > 1.5 * pitch_pix          # % is meaningless very close to the centre
ax.plot(r_meas[ok], dist_pct_meas[ok], '.', label='measured dots')
rr = np.linspace(0, corner_r, 200)
ax.plot(rr * rNorm, 100 * (k1 * rr**2 + k2 * rr**4), 'r-', label=f'model  k1={k1:.3e}, k2={k2:.3e}')
ax.axvline(r_meas.max(), color='gray', ls='--', lw=0.8, label='farthest dot')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('distance to distortion centre [pix]')
ax.set_ylabel('distortion [%]')
ax.set_title('Distortion curve (relative to paraxial scale)')
ax.grid(alpha=0.3)
ax.legend()
fig.savefig(os.path.join(out_dir, '5_distortion_curve.png'), dpi=150, bbox_inches='tight')
figs.append(fig)

# residuals after the radial model (non-radial errors)
fig, ax = plt.subplots(figsize=(9, 8))
cf = vector_map(ax, measuredPoints, res_rad,
                f'Residuals after radial model (non-radial errors: target tilt, noise...)\n'
                f'RMS {rms(res_rad_norm):.3f} pix, max {res_rad_norm.max():.3f} pix')
fig.colorbar(cf, ax=ax, label='|residual| [pix]', format='%.2f')
fig.savefig(os.path.join(out_dir, '6_residuals_after_radial.png'), dpi=150, bbox_inches='tight')
figs.append(fig)

# %% PDF report
fig = plt.figure(figsize=(8.27, 11.69))           # A4
fig.text(0.06, 0.96, summary, family='monospace', fontsize=8.5, va='top')
report_path = os.path.join(out_dir, 'FOVCAL_report.pdf')
with PdfPages(report_path) as pdf:
    for f in [fig] + figs:
        pdf.savefig(f, bbox_inches='tight')

# %% Save distortion file (one line per dot)
header = ('col,row,nominal_x_mm,nominal_y_mm,measured_x_pix,measured_y_pix,'
          'fit_x_pix,fit_y_pix,residual_x_pix,residual_y_pix,residual_norm_pix,'
          'radial_model_residual_x_pix,radial_model_residual_y_pix,distortion_pct,dot_radius_pix')
table = np.column_stack([ij_round, nominalPoints, measuredPoints, measFit,
                         residuals, res_norm, res_rad, dist_pct_meas, radii_filt])
np.savetxt(os.path.join(out_dir, 'distortion_points.csv'), table, delimiter=',',
           header=header, comments='', fmt='%.5f')

print(f'Results saved in ./{out_dir}/  (report: {report_path})')
plt.show()
