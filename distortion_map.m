function distortion_map(grid_img_path)
% DISTORTION_MAP  Classical distortion map of a lens from an image of a dot grid.
%
%   distortion_map('Grid.tif')
%
% Detects the dots, fits a radial distortion model
%   measured = c + (p - c)(1 + k1 r^2 + k2 r^4),  p = linear map of the nominal grid
% and plots the distortion (relative to the paraxial scale) over the full sensor.
% Base MATLAB only (no toolbox). Also runs in GNU Octave.

if nargin < 1
    grid_img_path = 'Grid.tif';
end

%% Parameters
radiusRange_pixs = [24 85];     % accepted dot radius range [pix]
dotSpacing_mm = 0.8;            % grid pitch on the target [mm]

%% Load image (grey levels)
[I, map] = imread(grid_img_path);
I = double(I);
if ~isempty(map)
    I = reshape(255 * mean(map(I(:) + 1, :), 2), size(I));
elseif size(I, 3) == 3
    I = mean(I, 3);
end
[resY, resX] = size(I);

%% Detect dots: Otsu threshold + regions + weighted centroids
mn = min(I(:));
mx = max(I(:));
bin = min(floor((I(:) - mn) / (mx - mn) * 256) + 1, 256);
h = accumarray(bin, 1, [256 1]);
mids = mn + ((1:256)' - 0.5) * (mx - mn) / 256;
w0 = cumsum(h);
w1 = w0(end) - w0;
c = cumsum(h .* mids);
[~, k] = max(w0 .* w1 .* (c ./ max(w0, 1) - (c(end) - c) ./ max(w1, 1)) .^ 2);
mask = I > mids(k);

Lb = label_regions(~mask);                      % fill dust specks inside the dots
border = unique([Lb(1, :), Lb(end, :), Lb(:, 1)', Lb(:, end)']);
mask = mask | (Lb > 0 & ~ismember(Lb, border));
[L, nLab] = label_regions(mask);

W = max(I - median(I(~mask)), 0);
[yy, xx] = ndgrid(1:resY, 1:resX);
in = L(:) > 0;
lab = L(in);
sw = accumarray(lab, W(in), [nLab 1]);
pts = [accumarray(lab, W(in) .* xx(in), [nLab 1]) ./ sw, ...
       accumarray(lab, W(in) .* yy(in), [nLab 1]) ./ sw];
radii = sqrt(accumarray(lab, 1, [nLab 1]) / pi);
cut = accumarray(lab, xx(in), [nLab 1], @min) == 1 | accumarray(lab, yy(in), [nLab 1], @min) == 1 | ...
      accumarray(lab, xx(in), [nLab 1], @max) == resX | accumarray(lab, yy(in), [nLab 1], @max) == resY;

keep = radii >= radiusRange_pixs(1) & radii <= radiusRange_pixs(2) & ~cut;
keep = keep & abs(radii - median(radii(keep))) < 0.2 * median(radii(keep));
pts = pts(keep, :);
n = size(pts, 1);

%% Grid indices from the two lattice vectors
dx = pts(:, 1)' - pts(:, 1);
dy = pts(:, 2)' - pts(:, 2);
dist = hypot(dx, dy);
dist(1:n + 1:end) = Inf;
pitch = median(min(dist, [], 2));
nn = dist > 0.7 * pitch & dist < 1.3 * pitch;
ang = atan2(dy(nn), dx(nn));
ix = abs(ang) < pi / 4;
iy = ang > pi / 4 & ang < 3 * pi / 4;
nnx = dx(nn);
nny = dy(nn);
B = [median(nnx(ix)), median(nnx(iy)); median(nny(ix)), median(nny(iy))];
[~, i0] = min(hypot(pts(:, 1) - resX / 2, pts(:, 2) - resY / 2));
ij = round((B \ (pts - pts(i0, :))')');
nominal = (ij - min(ij, [], 1)) * dotSpacing_mm;

%% Radial distortion fit
rNorm = hypot(resX, resY) / 2;
T = [nominal, ones(n, 1)] \ pts;                % linear start value (3x2)
p0 = [resX / 2; resY / 2; 0; 0; reshape(T', [], 1)];
p = lm_fit(@(p) reshape(radial_model(p, nominal, rNorm) - pts, [], 1), p0);
cx = p(1);
cy = p(2);
k1 = p(3);
k2 = p(4);

r_dots = max(hypot(pts(:, 1) - cx, pts(:, 2) - cy)) / rNorm;
corner_r = max(hypot([1 resX 1 resX] - cx, [1 1 resY resY] - cy)) / rNorm;
pct = @(r) 100 * (k1 * r .^ 2 + k2 * r .^ 4);

%% Plot
figure('Name', 'Distortion map', 'Color', 'w', 'Position', [80 80 900 850]);
ax = gca;
hold(ax, 'on');
[gx, gy] = meshgrid(linspace(1, resX, 300), linspace(1, resY, 300));
gd = pct(hypot(gx - cx, gy - cy) / rNorm);
contourf(ax, gx, gy, gd, 20, 'LineStyle', 'none');
[C, hc] = contour(ax, gx, gy, gd, 8, 'LineColor', 'w');
clabel(C, hc, 'Color', 'w', 'FontSize', 7);

[vx, vy] = meshgrid(linspace(64, resX - 64, 15), linspace(64, resY - 64, 15));
f = pct(hypot(vx - cx, vy - cy) / rNorm) / 100;
ddx = (vx - cx) .* f;
ddy = (vy - cy) .* f;
dmax = max(hypot(ddx(:), ddy(:)));
s = 0.8 * (resX / 14) / dmax;                   % arrow magnification
quiver(ax, vx, vy, ddx * s, ddy * s, 0, 'k');
key = str2double(sprintf('%.1g', dmax / 2));    % reference arrow
quiver(ax, 0.03 * resX, 0.975 * resY, key * s, 0, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 1);
text(ax, 0.03 * resX + key * s + 20, 0.975 * resY, sprintf('%g pix', key), ...
    'Color', 'r', 'BackgroundColor', 'w', 'VerticalAlignment', 'middle');

h1 = plot(ax, [min(pts(:, 1)) max(pts(:, 1)) max(pts(:, 1)) min(pts(:, 1)) min(pts(:, 1))], ...
              [min(pts(:, 2)) min(pts(:, 2)) max(pts(:, 2)) max(pts(:, 2)) min(pts(:, 2))], 'w--');
h2 = plot(ax, cx, cy, 'r+', 'MarkerSize', 14, 'LineWidth', 2);
h3 = plot(ax, (resX + 1) / 2, (resY + 1) / 2, 'wx', 'MarkerSize', 8, 'LineWidth', 1.5);
lg = legend([h1 h2 h3], {'area covered by dots', 'distortion centre', 'image centre'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal');
set(lg, 'Color', [0.45 0.45 0.45], 'TextColor', 'w');

axis(ax, 'equal');
xlim(ax, [1 resX]);
ylim(ax, [1 resY]);
set(ax, 'YDir', 'reverse', 'Box', 'on');
xlabel(ax, 'x [pix]');
ylabel(ax, 'y [pix]');
title(ax, {'Classical distortion map (relative to paraxial scale)', ...
    sprintf('%+.3f %% at farthest dot, %+.3f %% at image corner   (k1 = %.3e, k2 = %.3e)', ...
            pct(r_dots), pct(corner_r), k1, k2)});
cb = colorbar(ax);
try
    cb.Label.String = 'distortion [%]';         % MATLAB
catch
    ylabel(cb, 'distortion [%]');               % Octave
end
print(gcf, 'distortion_map.png', '-dpng', '-r150');
end


%% ---------------------------------------------------------------------------
function [L, nLab] = label_regions(mask)
% 4-connected labelling of a logical image (replacement for bwlabel).
[nr, nc] = size(mask);
d = diff([false(1, nc); mask; false(1, nc)]);
[rs, cs] = find(d == 1);                        % run starts (row, col)
re = mod(find(d == -1) - 1, nr + 1);            % run ends (last true row)
nRuns = numel(rs);
parent = 1:nRuns;
colStart = [0; cumsum(accumarray(cs, 1, [nc 1]))];
for col = 2:nc
    a = colStart(col - 1) + 1;
    aEnd = colStart(col);
    b = colStart(col) + 1;
    bEnd = colStart(col + 1);
    while a <= aEnd && b <= bEnd
        if rs(a) <= re(b) && rs(b) <= re(a)     % overlapping runs -> same region
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
    % floor: with k1 = k2 = 0 the model does not depend on the centre
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
