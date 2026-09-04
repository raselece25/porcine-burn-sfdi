function Rd = diffuseReflectance(mua, musp, fx, nTissue)
%DIFFUSEREFLECTANCE Diffusion-approximation SFDI forward model.
%   RD = DIFFUSEREFLECTANCE(MUA, MUSP, FX, NTISSUE) predicts the diffuse
%   reflectance of a semi-infinite homogeneous turbid medium at spatial
%   frequency FX [mm^-1], given absorption coefficient MUA [mm^-1] and
%   reduced scattering coefficient MUSP [mm^-1]. NTISSUE (default 1.4)
%   sets the tissue refractive index used for the internal-reflectance
%   boundary condition.
%
%   Reference: Cuccia et al., "Quantitation and mapping of tissue
%   optical properties using modulated imaging," J. Biomed. Opt. 14(2),
%   024012 (2009).
%
%   This is a MATLAB port of src/diffusion_model.py in this repository.

    if nargin < 4
        nTissue = 1.4;
    end

    n = nTissue;
    rD = -1.440 * n^-2 + 0.710 * n^-1 + 0.668 + 0.0636 * n;
    A = (1 + rD) / (1 - rD);

    muTr = mua + musp;
    aPrime = musp ./ muTr;

    muEffFx = sqrt(3 .* mua .* muTr + (2 * pi * fx)^2);

    ratio = muEffFx ./ muTr;
    Rd = (3 * A .* aPrime) ./ ((ratio + 1) .* (ratio + 3 * A));
end
