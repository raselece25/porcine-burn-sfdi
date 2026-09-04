function [muaMap, muspMap] = invertOpticalProperties(RdMeasured, fxList, muaGrid, muspGrid, nTissue)
%INVERTOPTICALPROPERTIES Lookup-table inversion of calibrated SFDI data.
%   [MUAMAP, MUSPMAP] = INVERTOPTICALPROPERTIES(RDMEASURED, FXLIST,
%   MUAGRID, MUSPGRID, NTISSUE) recovers per-pixel (mu_a, mu_s') maps
%   from calibrated diffuse-reflectance images at two or more spatial
%   frequencies.
%
%   RDMEASURED : cell array of 2-D images, one per entry in FXLIST
%   FXLIST     : vector of spatial frequencies [mm^-1] matching RDMEASURED
%   MUAGRID, MUSPGRID : 1-D search grids [mm^-1]
%   NTISSUE    : tissue refractive index (default 1.4)
%
%   This performs an exhaustive nearest-match search over the
%   MUAGRID x MUSPGRID forward-model lookup table -- adequate for small
%   demo images; a production pipeline would use a finer/adaptive grid
%   or a gradient-based solver. See src/diffusion_model.py for the
%   equivalent (and more thoroughly documented) Python implementation.

    if nargin < 5
        nTissue = 1.4;
    end

    [imRows, imCols] = size(RdMeasured{1});
    nFx = numel(fxList);

    [muaMesh, muspMesh] = ndgrid(muaGrid, muspGrid);
    lut = zeros(numel(muaGrid), numel(muspGrid), nFx);
    for k = 1:nFx
        lut(:, :, k) = diffuseReflectance(muaMesh, muspMesh, fxList(k), nTissue);
    end
    lutFlat = reshape(lut, [], nFx)';           % nFx x (nMua*nMusp)
    muaFlatGrid = reshape(muaMesh, 1, []);
    muspFlatGrid = reshape(muspMesh, 1, []);

    muaMap = zeros(imRows, imCols);
    muspMap = zeros(imRows, imCols);

    for r = 1:imRows
        for c = 1:imCols
            target = zeros(nFx, 1);
            for k = 1:nFx
                target(k) = RdMeasured{k}(r, c);
            end
            sqErr = sum((lutFlat - target).^2, 1);
            [~, bestIdx] = min(sqErr);
            muaMap(r, c) = muaFlatGrid(bestIdx);
            muspMap(r, c) = muspFlatGrid(bestIdx);
        end
    end
end
