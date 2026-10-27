function R = fovcal(grid_img_path)
% FOVCAL  FOV calibration from an image of a dot grid target.
%
%   R = fovcal('Grid.tif')
%
% The nominal grid is fitted to the measured dot positions with a similarity
% (translation, rotation, scale) or affine transform. The fit scale gives the
% camera scale and, with the sensor pixel pitch, the magnification. The fit
% residuals contain the nonlinear lens aberrations, including distortion.
% A radial distortion model (k1, k2, free centre) is fitted as well.
%
% Base MATLAB only: no toolbox needed (no imfindcircles, fitgeotrans,
% bwlabel, lsqnonlin...). Also runs in GNU Octave.
%
% Outputs in out_dir: FOVCAL_report.pdf, PNG figures, distortion_points.csv,
% summary.txt. R is a struct with all the results.
%
% Pixel coordinates are MATLAB 1-based (centre of the top-left pixel = (1, 1)).

if nargin < 1
    grid_img_path = 'Grid.tif';
end

%% Parameters
radiusRange_pixs = [24 85];     % accepted dot radius range [pix]
dotSpacing_mm = 0.8;            % grid pitch on the target [mm]
pixelSize_mm = 5.5e-3;          % camera pixel size [mm]
% 'similarity': translation + rotation + one scale (4 params, FOVCAL definition)
% 'affine'    : + separate x/y scale and shear (6 params, like fitgeotrans 'affine')
fit_type = 'similarity';
out_dir = 'results_matlab';

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% Load image (grey levels, as double)
[I, map] = imread(grid_img_path);
I = double(I);
if ~isempty(map)                        % indexed image -> grey levels via the palette
    I = reshape(255 * mean(map(I(:) + 1, :), 2), size(I));
elseif size(I, 3) == 3                  % RGB -> grey
    I = 0.2989 * I(:,:,1) + 0.5870 * I(:,:,2) + 0.1140 * I(:,:,3);
end
[resY, resX] = size(I);

%% Detect dots (bright blobs)
% Otsu threshold
mn = min(I(:));
mx = max(I(:));
bin = min(floor((I(:) - mn) / (mx - mn) * 256) + 1, 256);
h = accumarray(bin, 1, [256 1]);
mids = mn + ((1:256)' - 0.5) * (mx - mn) / 256;
w0 = cumsum(h);
w1 = w0(end) - w0;
c = cumsum(h .* mids);
m0 = c ./ max(w0, 1);
m1 = (c(end) - c) ./ max(w1, 1);
[~, k] = max(w0 .* w1 .* (m0 - m1) .^ 2);
thr = mids(k);

mask = I > thr;
% fill dust specks inside the dots: background regions not touching the border
Lb = label_regions(~mask);
border = unique([Lb(1, :), Lb(end, :), Lb(:, 1)', Lb(:, end)']);
mask = mask | (Lb > 0 & ~ismember(Lb, border));
[L, nLab] = label_regions(mask);

% intensity-weighted centroid (background removed) -> sub-pixel centres
bg = median(I(~mask));
W = max(I - bg, 0);
[yy, xx] = ndgrid(1:resY, 1:resX);
in = L(:) > 0;
lab = L(in);
sw = accumarray(lab, W(in), [nLab 1]);
centers = [accumarray(lab, W(in) .* xx(in), [nLab 1]) ./ sw, ...
           accumarray(lab, W(in) .* yy(in), [nLab 1]) ./ sw];      % (x, y)
areas = accumarray(lab, 1, [nLab 1]);
radii = sqrt(areas / pi);
touches_border = accumarray(lab, xx(in), [nLab 1], @min) == 1 | ...
                 accumarray(lab, yy(in), [nLab 1], @min) == 1 | ...
                 accumarray(lab, xx(in), [nLab 1], @max) == resX | ...
                 accumarray(lab, yy(in), [nLab 1], @max) == resY;
clear yy xx in lab W Lb

%% Filter dots
% 1) radius in range, 2) not cut by the border
keep = radii >= radiusRange_pixs(1) & radii <= radiusRange_pixs(2) & ~touches_border;
% 3) margin of one mean radius from the border (as in the original script)
margin = mean(radii(keep));
keep = keep & centers(:, 1) > margin & centers(:, 1) < resX - margin & ...
              centers(:, 2) > margin & centers(:, 2) < resY - margin;
% 4) reject size outliers (merged / partial blobs)
medR = median(radii(keep));
keep = keep & abs(radii - medR) < 0.2 * medR;

centers_filt = centers(keep, :);
radii_filt = radii(keep);
fprintf('Dots detected: %d, kept after filtering: %d\n', nLab, size(centers_filt, 1));

%% Show filtered centres
fig1 = figure('Name', 'Detected dots', 'Color', 'w', 'Position', [50 50 800 800]);
imagesc(I);
colormap(gray);
axis image;
hold on;
t = linspace(0, 2 * pi, 41);
cxs = [centers_filt(:, 1) + radii_filt .* cos(t), nan(numel(radii_filt), 1)]';
cys = [centers_filt(:, 2) + radii_filt .* sin(t), nan(numel(radii_filt), 1)]';
plot(cxs(:), cys(:), 'g-');
plot(centers_filt(:, 1), centers_filt(:, 2), 'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
title('Filtered circle centers');
hold off;

%% Assign grid indices (i = column, j = row)
% Estimate the two lattice vectors from nearest-neighbour displacements.
% This replaces the "sort into rows" step, which breaks when the grid is
% slightly rotated or when a dot is missing at the start of a row.
pts = centers_filt;
n = size(pts, 1);
dx = pts(:, 1)' - pts(:, 1);            % dx(i, j) = x_j - x_i
dy = pts(:, 2)' - pts(:, 2);
dist = hypot(dx, dy);
dist(1:n + 1:end) = Inf;
pitch_pix = median(min(dist, [], 2));

nn = dist > 0.7 * pitch_pix & dist < 1.3 * pitch_pix;      % all 1st-neighbour vectors
nnx = dx(nn);
nny = dy(nn);
ang = atan2(nny, nnx);
ix = abs(ang) < pi / 4;                 % vector along x
iy = ang > pi / 4 & ang < 3 * pi / 4;   % vector along y
B = [median(nnx(ix)), median(nnx(iy));
     median(nny(ix)), median(nny(iy))];

% origin = dot closest to the image centre, then integer coordinates
[~, i0] = min(hypot(pts(:, 1) - resX / 2, pts(:, 2) - resY / 2));
ij = (B \ (pts - pts(i0, :))')';
ij_round = round(ij);
if max(abs(ij(:) - ij_round(:))) > 0.25
    warning('Some dots are far from the ideal lattice, check figure 1');
end
ij_round = ij_round - min(ij_round, [], 1);   % start indices at 0 (top-left dot)

nominalPoints = ij_round * dotSpacing_mm;     % [mm]  (x right, y down, like the image)
measuredPoints = pts;                         % [pix]
nCols = max(ij_round(:, 1)) + 1;
nRows = max(ij_round(:, 2)) + 1;
fprintf('Grid: %d columns x %d rows\n', nCols, nRows);

%% Fit transform  (nominal [mm] -> measured [pix])
% measured = [x_mm, y_mm, 1] * T   (same convention as fitgeotrans: tform.T)
X = [nominalPoints, ones(n, 1)];
switch fit_type
    case 'affine'
        T = X \ measuredPoints;                                 % 3x2
    case 'similarity'
        % x' = a x - b y + tx,  y' = b x + a y + ty   with a = s cos(th), b = s sin(th)
        nx = nominalPoints(:, 1);
        ny = nominalPoints(:, 2);
        M = [nx, -ny, ones(n, 1), zeros(n, 1);
             ny,  nx, zeros(n, 1), ones(n, 1)];
        s4 = M \ [measuredPoints(:, 1); measuredPoints(:, 2)];
        T = [s4(1), s4(2); -s4(2), s4(1); s4(3), s4(4)];
    otherwise
        error('unknown fit_type: %s', fit_type);
end
T = [T, [0; 0; 1]];                                             % 3x3 like fitgeotrans

%% Compute fit and residuals
measFit = X * T(:, 1:2);
residuals = measuredPoints - measFit;                           % [pix]
res_norm = hypot(residuals(:, 1), residuals(:, 2));

%% Magnification
A = T(1:2, 1:2);
scale_pix_per_mm = sqrt(abs(det(A)));
magnification = scale_pix_per_mm * pixelSize_mm;
sx = hypot(A(1, 1), A(1, 2));       % scale along grid x [pix/mm]
sy = hypot(A(2, 1), A(2, 2));       % scale along grid y [pix/mm]
rot_deg = atan2(A(1, 2), A(1, 1)) * 180 / pi;

%% Radial distortion fit (Brown model, distortion centre free)
% measured = c + (p - c) * (1 + k1 r^2 + k2 r^4),  p = linear-mapped nominal point,
% r normalised by the image half-diagonal so k1, k2 are dimensionless.
% The linear part p is the scale at the distortion centre (paraxial scale): it does
% not depend on which part of the field is covered by dots.
rNorm = hypot(resX, resY) / 2;
p0 = [resX / 2; resY / 2; 0; 0; reshape(T(:, 1:2)', [], 1)];
resfun = @(p) reshape(radial_model(p, nominalPoints, rNorm) - measuredPoints, [], 1);
p = lm_fit(resfun, p0);
cx = p(1);
cy = p(2);
k1 = p(3);
k2 = p(4);
T6 = reshape(p(5:10), 2, 3)';
res_rad = measuredPoints - radial_model(p, nominalPoints, rNorm);
res_rad_norm = hypot(res_rad(:, 1), res_rad(:, 2));

scale_par = sqrt(abs(det(T6(1:2, :))));
mag_par = scale_par * pixelSize_mm;

% classical distortion of each dot [%] = (real radius - paraxial radius) / paraxial radius
p_lin = X * T6;
r_lin = hypot(p_lin(:, 1) - cx, p_lin(:, 2) - cy);
r_meas = hypot(measuredPoints(:, 1) - cx, measuredPoints(:, 2) - cy);
dist_pct_meas = 100 * (r_meas - r_lin) ./ r_lin;

% distortion in % at the farthest dot and at the image corner (model)
r_dots = max(r_meas) / rNorm;
dist_pct_dots = 100 * (k1 * r_dots ^ 2 + k2 * r_dots ^ 4);
corner_r = max(hypot([1 resX 1 resX] - cx, [1 1 resY resY] - cy)) / rNorm;
dist_pct_corner = 100 * (k1 * corner_r ^ 2 + k2 * corner_r ^ 4);

rms = @(v) sqrt(mean(v .^ 2));
um_per_pix = 1e3 / scale_pix_per_mm;
if dist_pct_dots < 0
    dist_kind = 'barrel';
else
    dist_kind = 'pincushion';
end

%% Summary (console, summary.txt and first page of the PDF report)
sep1 = repmat('=', 1, 60);
sep2 = repmat('-', 1, 60);
S = {
    'FOV CALIBRATION REPORT'
    sep1
    sprintf('Image                       : %s  (%d x %d pix)', grid_img_path, resX, resY)
    sprintf('Date                        : %s', datestr(now, 'yyyy-mm-dd HH:MM'))
    sprintf('Dot spacing (target)        : %g mm', dotSpacing_mm)
    sprintf('Sensor pixel size           : %.2f um', pixelSize_mm * 1e3)
    sprintf('Dots used                   : %d  (%d columns x %d rows)', n, nCols, nRows)
    ''
    sprintf('1) FOVCAL FIT  (nominal grid -> measured grid, %s)', fit_type)
    sep2
    'Transform matrix (fitgeotrans tform.T convention):'
    sprintf('  [%12.5f %12.5f %4g]', T(1, :))
    sprintf('  [%12.5f %12.5f %4g]', T(2, :))
    sprintf('  [%12.5f %12.5f %4g]', T(3, :))
    sprintf('Scale                       : %.3f pix/mm   (x %.3f, y %.3f)', scale_pix_per_mm, sx, sy)
    sprintf('MAGNIFICATION               : %.5f   (x %.5f, y %.5f)', magnification, sx * pixelSize_mm, sy * pixelSize_mm)
    sprintf('Pixel size on object        : %.4f um/pix', um_per_pix)
    sprintf('Field of view on object     : %.3f x %.3f mm', resX * um_per_pix / 1e3, resY * um_per_pix / 1e3)
    sprintf('Grid rotation               : %.4f deg', rot_deg)
    'Residuals (= lens aberrations, incl. distortion)'
    sprintf('   RMS                      : %.3f pix  = %.2f um on object', rms(res_norm), rms(res_norm) * um_per_pix)
    sprintf('   max                      : %.3f pix  = %.2f um on object', max(res_norm), max(res_norm) * um_per_pix)
    ''
    '2) RADIAL DISTORTION MODEL  (measured = c + (p - c)(1 + k1 r^2 + k2 r^4))'
    sep2
    sprintf('Distortion centre (x, y)    : (%.1f, %.1f) pix', cx, cy)
    sprintf('   offset from image centre : (%+.1f, %+.1f) pix', cx - (resX + 1) / 2, cy - (resY + 1) / 2)
    sprintf('k1, k2  (r norm. %.1f pix): %.4e, %.4e', rNorm, k1, k2)
    sprintf('Paraxial magnification      : %.5f   (scale at the distortion centre)', mag_par)
    sprintf('Distortion at farthest dot  : %+.4f %%   (%s)', dist_pct_dots, dist_kind)
    sprintf('Distortion at image corner  : %+.4f %%   (extrapolated)', dist_pct_corner)
    sprintf('Residuals after radial model: RMS %.3f pix, max %.3f pix', rms(res_rad_norm), max(res_rad_norm))
    '   (what is left: non-radial errors such as target tilt, target errors, noise)'
    ''
    'HOW TO READ THE VECTOR MAPS'
    sep2
    '- FOVCAL residual map: measured - fitted position. The fit takes the'
    '  average scale over the dots, so with pincushion distortion the vectors'
    '  point inward in the middle and outward in the corners (barrel: reversed).'
    '- Classical distortion map: position relative to the paraxial (centre)'
    '  scale. Pure radial distortion gives vectors pointing away from the'
    '  centre (pincushion) or towards it (barrel), growing with the radius.'
    '- Arrows are magnified; the red key arrow gives the arrow length.'
    };
fprintf('\n');
fprintf('%s\n', S{:});
fid = fopen(fullfile(out_dir, 'summary.txt'), 'w');
fprintf(fid, '%s\n', S{:});
fclose(fid);

%% Report figures
G = struct('resX', resX, 'resY', resY, 'pitch_pix', pitch_pix, 'um_per_pix', um_per_pix);
figs = {fig1};
names = {'1_detected_dots'};

% FOVCAL residual vector field
f = figure('Name', 'FOVCAL residuals', 'Color', 'w', 'Position', [60 60 900 800]);
vector_map(gca, measuredPoints, residuals, G, ...
    {sprintf('FOVCAL residuals (%s fit) = lens aberrations', fit_type), ...
     sprintf('RMS %.3f pix, max %.3f pix', rms(res_norm), max(res_norm))});
figs{end + 1} = f;
names{end + 1} = '2_residual_vectors';

% x / y components of the residuals
f = figure('Name', 'Residual components', 'Color', 'w', 'Position', [70 70 1100 600]);
component_maps(measuredPoints, residuals, G, 'FOVCAL residual');
figs{end + 1} = f;
names{end + 1} = '3_residual_components';

% classical distortion field over the full sensor (radial model)
f = figure('Name', 'Distortion map', 'Color', 'w', 'Position', [80 80 900 850]);
ax = gca;
hold(ax, 'on');
[gx, gy] = meshgrid(linspace(1, resX, 300), linspace(1, resY, 300));
gr2 = ((gx - cx) .^ 2 + (gy - cy) .^ 2) / rNorm ^ 2;
gd = 100 * (k1 * gr2 + k2 * gr2 .^ 2);
contourf(ax, gx, gy, gd, 20, 'LineStyle', 'none');
[C, hc] = contour(ax, gx, gy, gd, 8, 'LineColor', 'w');
clabel(C, hc, 'Color', 'w', 'FontSize', 7);
[vx, vy] = meshgrid(linspace(64, resX - 64, 15), linspace(64, resY - 64, 15));
vr2 = ((vx - cx) .^ 2 + (vy - cy) .^ 2) / rNorm ^ 2;
ddx = (vx - cx) .* (k1 * vr2 + k2 * vr2 .^ 2);
ddy = (vy - cy) .* (k1 * vr2 + k2 * vr2 .^ 2);
dmax = max(hypot(ddx(:), ddy(:)));
s = 0.8 * (resX / 14) / dmax;
quiver(ax, vx, vy, ddx * s, ddy * s, 0, 'k');
draw_key(ax, dmax, s, G);
x0 = min(measuredPoints(:, 1));
x1 = max(measuredPoints(:, 1));
y0 = min(measuredPoints(:, 2));
y1 = max(measuredPoints(:, 2));
h1 = plot(ax, [x0 x1 x1 x0 x0], [y0 y0 y1 y1 y0], 'w--', 'LineWidth', 1);
h2 = plot(ax, cx, cy, 'r+', 'MarkerSize', 14, 'LineWidth', 2);
h3 = plot(ax, (resX + 1) / 2, (resY + 1) / 2, 'wx', 'MarkerSize', 8, 'LineWidth', 1.5);
lg = legend([h1 h2 h3], {'area covered by dots', 'distortion centre', 'image centre'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal');
set(lg, 'Color', [0.45 0.45 0.45], 'TextColor', 'w');
set_image_axes(ax, G);
title(ax, {'Classical distortion map (radial model, relative to paraxial scale)', ...
    sprintf('%+.3f %% at farthest dot, %+.3f %% at image corner', dist_pct_dots, dist_pct_corner)});
cb_label(colorbar(ax), 'distortion [%]', false);
figs{end + 1} = f;
names{end + 1} = '4_distortion_map';

% distortion curve
f = figure('Name', 'Distortion curve', 'Color', 'w', 'Position', [90 90 900 550]);
ok = r_lin > 1.5 * pitch_pix;          % % is meaningless very close to the centre
plot(r_meas(ok), dist_pct_meas(ok), '.', 'MarkerSize', 10);
hold on;
rr = linspace(0, corner_r, 200);
plot(rr * rNorm, 100 * (k1 * rr .^ 2 + k2 * rr .^ 4), 'r-', 'LineWidth', 1.5);
plot([1 1] * max(r_meas), ylim, '--', 'Color', [0.5 0.5 0.5]);
plot(xlim, [0 0], 'k-', 'LineWidth', 0.5);
grid on;
xlabel('distance to distortion centre [pix]');
ylabel('distortion [%]');
title('Distortion curve (relative to paraxial scale)');
legend({'measured dots', sprintf('model  k1=%.3e, k2=%.3e', k1, k2), 'farthest dot'}, ...
    'Location', 'northwest');
hold off;
figs{end + 1} = f;
names{end + 1} = '5_distortion_curve';

% residuals after the radial model (non-radial errors)
f = figure('Name', 'Residuals after radial model', 'Color', 'w', 'Position', [100 100 900 800]);
vector_map(gca, measuredPoints, res_rad, G, ...
    {'Residuals after radial model (non-radial errors: target tilt, noise...)', ...
     sprintf('RMS %.3f pix, max %.3f pix', rms(res_rad_norm), max(res_rad_norm))});
figs{end + 1} = f;
names{end + 1} = '6_residuals_after_radial';

%% PDF report + PNG figures
fs = figure('Name', 'Summary', 'Color', 'w', 'Position', [40 40 827 1169]);
axes('Position', [0 0 1 1], 'Visible', 'off');
text(0.05, 0.97, S, 'Units', 'normalized', 'FontName', 'Courier New', 'FontSize', 8, ...
    'Interpreter', 'none', 'VerticalAlignment', 'top');

report_path = fullfile(out_dir, 'FOVCAL_report.pdf');
if exist(report_path, 'file')
    delete(report_path);
end
all_figs = [{fs}, figs];
for kf = 1:numel(all_figs)
    append_pdf(all_figs{kf}, report_path);
end
for kf = 1:numel(figs)
    print(figs{kf}, fullfile(out_dir, [names{kf} '.png']), '-dpng', '-r150');
end

%% Save distortion file (one line per dot)
fid = fopen(fullfile(out_dir, 'distortion_points.csv'), 'w');
fprintf(fid, ['col,row,nominal_x_mm,nominal_y_mm,measured_x_pix,measured_y_pix,' ...
    'fit_x_pix,fit_y_pix,residual_x_pix,residual_y_pix,residual_norm_pix,' ...
    'radial_model_residual_x_pix,radial_model_residual_y_pix,distortion_pct,dot_radius_pix\n']);
tab = [ij_round, nominalPoints, measuredPoints, measFit, residuals, res_norm, ...
       res_rad, dist_pct_meas, radii_filt];
fprintf(fid, [repmat('%.5f,', 1, size(tab, 2) - 1) '%.5f\n'], tab');
fclose(fid);

fprintf('\nResults saved in %s  (report: %s)\n', out_dir, report_path);

%% Results struct
R = struct('T', T, 'fit_type', fit_type, 'scale_pix_per_mm', scale_pix_per_mm, ...
    'scale_x', sx, 'scale_y', sy, 'magnification', magnification, ...
    'rotation_deg', rot_deg, 'um_per_pix', um_per_pix, ...
    'residual_rms_pix', rms(res_norm), 'residual_max_pix', max(res_norm), ...
    'distortion_centre', [cx cy], 'k1', k1, 'k2', k2, 'paraxial_magnification', mag_par, ...
    'distortion_pct_farthest_dot', dist_pct_dots, 'distortion_pct_corner', dist_pct_corner, ...
    'nominalPoints', nominalPoints, 'measuredPoints', measuredPoints, ...
    'residuals', residuals, 'grid_index', ij_round);
end


%% ---------------------------------------------------------------------------
function [L, nLab] = label_regions(mask)
% 4-connected labelling of a logical image (replacement for bwlabel).
% Works on vertical runs of true pixels, merged with a union-find.
[nr, nc] = size(mask);
d = diff([false(1, nc); mask; false(1, nc)]);
[rs, cs] = find(d == 1);                % run starts (row, col), column-major order
re = find(d == -1);
re = mod(re - 1, nr + 1);               % run ends (last true row)
nRuns = numel(rs);
parent = 1:nRuns;
colStart = [0; cumsum(accumarray(cs, 1, [nc 1]))];   % runs of column c: colStart(c)+1..colStart(c+1)

for col = 2:nc
    a = colStart(col - 1) + 1;          % runs of the previous column
    aEnd = colStart(col);
    b = colStart(col) + 1;              % runs of this column
    bEnd = colStart(col + 1);
    while a <= aEnd && b <= bEnd
        if rs(a) <= re(b) && rs(b) <= re(a)     % overlapping -> same region
            ra = find_root(parent, a);
            rb = find_root(parent, b);
            parent(max(ra, rb)) = min(ra, rb);
        end
        if re(a) < re(b)
            a = a + 1;
        else
            b = b + 1;
        end
    end
end

roots = zeros(nRuns, 1);
for kr = 1:nRuns
    roots(kr) = find_root(parent, kr);
end
[~, ~, runLab] = unique(roots);
nLab = max([runLab; 0]);

L = zeros(nr, nc);
for kr = 1:nRuns
    L(rs(kr):re(kr), cs(kr)) = runLab(kr);
end
end


function r = find_root(parent, k)
r = k;
while parent(r) ~= r
    r = parent(r);
end
end


function M = radial_model(p, P, rNorm)
% measured = c + (q - c)(1 + k1 r^2 + k2 r^4),  q = [P 1] * T6
T6 = reshape(p(5:10), 2, 3)';
q = ([P, ones(size(P, 1), 1)] * T6 - p(1:2)') / rNorm;
r2 = sum(q .^ 2, 2);
M = p(1:2)' + q * rNorm .* (1 + p(3) * r2 + p(4) * r2 .^ 2);
end


function p = lm_fit(resfun, p)
% Levenberg-Marquardt least squares with a numerical Jacobian (replaces lsqnonlin).
lambda = 1e-3;
r = resfun(p);
cost = r' * r;
for it = 1:200
    J = zeros(numel(r), numel(p));
    for kp = 1:numel(p)
        hstep = 1e-6 * max(abs(p(kp)), 1);
        pk = p;
        pk(kp) = pk(kp) + hstep;
        J(:, kp) = (resfun(pk) - r) / hstep;
    end
    JJ = J' * J;
    g = J' * r;
    % damping scaled per parameter; floor needed because with k1 = k2 = 0 the
    % model does not depend on the centre (zero Jacobian columns)
    D = diag(max(diag(JJ), 1e-9 * max(diag(JJ))));
    improved = false;
    converged = false;
    while lambda < 1e10
        pn = p - (JJ + lambda * D) \ g;
        rn = resfun(pn);
        cn = rn' * rn;
        if cn < cost
            converged = (cost - cn) < 1e-12 * cost;
            p = pn;
            r = rn;
            cost = cn;
            lambda = max(lambda / 10, 1e-12);
            improved = true;
            break;
        end
        lambda = lambda * 10;
    end
    if ~improved || converged
        break;
    end
end
end


function vector_map(ax, P, V, G, ttl)
% Vector field of V at points P, over an interpolated |V| colour map.
hold(ax, 'on');
mag = hypot(V(:, 1), V(:, 2));
[gx, gy] = meshgrid(linspace(1, G.resX, 300), linspace(1, G.resY, 300));
% interpolate the components, not |V|, so the colour map stays >= 0
gvx = griddata(P(:, 1), P(:, 2), V(:, 1), gx, gy, 'linear');
gvy = griddata(P(:, 1), P(:, 2), V(:, 2), gx, gy, 'linear');
contourf(ax, gx, gy, hypot(gvx, gvy), linspace(0, max(mag), 21), 'LineStyle', 'none');
caxis(ax, [0 max(mag)]);
s = 0.8 * G.pitch_pix / max(max(mag), 1e-9);          % arrow magnification
quiver(ax, P(:, 1), P(:, 2), V(:, 1) * s, V(:, 2) * s, 0, 'k');
draw_key(ax, max(mag), s, G);
set_image_axes(ax, G);
title(ax, ttl, 'Interpreter', 'none');
cb_label(colorbar(ax), '|residual| [pix]', false);
end


function component_maps(P, V, G, what)
lim = max(abs(V(:)));
[gx, gy] = meshgrid(linspace(1, G.resX, 300), linspace(1, G.resY, 300));
x = linspace(0, 1, 64)';
cmap = [interp1([0 0.5 1], [0.13 1 0.70], x), ...
        interp1([0 0.5 1], [0.40 1 0.09], x), ...
        interp1([0 0.5 1], [0.67 1 0.17], x)];        % blue - white - red
comp_names = {'x', 'y'};
for kc = 1:2
    ax = subplot(1, 2, kc);
    hold(ax, 'on');
    g = griddata(P(:, 1), P(:, 2), V(:, kc), gx, gy, 'linear');
    contourf(ax, gx, gy, g, linspace(-lim, lim, 21), 'LineStyle', 'none');
    plot(ax, P(:, 1), P(:, 2), 'k.', 'MarkerSize', 3);
    colormap(ax, cmap);
    caxis(ax, [-lim lim]);
    set_image_axes(ax, G);
    title(ax, sprintf('%s - %s component', what, comp_names{kc}));
    cb_label(colorbar(ax, 'southoutside'), sprintf('d%s [pix]', comp_names{kc}), true);
end
end


function draw_key(ax, vmax, s, G)
% reference arrow in the bottom-left corner
key = str2double(sprintf('%.1g', vmax / 2));
x0 = 0.03 * G.resX;
y0 = 0.975 * G.resY;
quiver(ax, x0, y0, key * s, 0, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 1);
text(ax, x0 + key * s + 20, y0, sprintf('%g pix (%.2g um)', key, key * G.um_per_pix), ...
    'Color', 'r', 'FontSize', 9, 'BackgroundColor', 'w', 'VerticalAlignment', 'middle');
end


function cb_label(cb, txt, horizontal)
try
    cb.Label.String = txt;              % MATLAB colorbar object
catch
    if horizontal                       % Octave: the colorbar is an axes
        xlabel(cb, txt);
    else
        ylabel(cb, txt);
    end
end
end


function set_image_axes(ax, G)
axis(ax, 'equal');
xlim(ax, [1 G.resX]);
ylim(ax, [1 G.resY]);
set(ax, 'YDir', 'reverse', 'Box', 'on');
xlabel(ax, 'x [pix]');
ylabel(ax, 'y [pix]');
end


function append_pdf(fig, pdf_path)
% one figure per page; exportgraphics (MATLAB R2021b+) or print -append (Octave)
if exist('OCTAVE_VERSION', 'builtin')
    if exist(pdf_path, 'file')
        print(fig, pdf_path, '-dpdf', '-append');
    else
        print(fig, pdf_path, '-dpdf');
    end
else
    try
        exportgraphics(fig, pdf_path, 'Append', true, 'Resolution', 150);
    catch
        % older MATLAB: one PDF per figure
        [d, nm] = fileparts(pdf_path);
        print(fig, fullfile(d, sprintf('%s_%d.pdf', nm, numel(dir(fullfile(d, [nm '_*.pdf']))) + 1)), ...
            '-dpdf', '-bestfit');
    end
end
end
