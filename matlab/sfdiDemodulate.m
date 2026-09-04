function [AC, DC] = sfdiDemodulate(I1, I2, I3)
%SFDIDEMODULATE Three-phase SFDI demodulation.
%   [AC, DC] = SFDIDEMODULATE(I1, I2, I3) recovers the AC (modulated) and
%   DC (planar) reflectance components from three images captured with
%   the same spatial-frequency illumination pattern phase-shifted by
%   0, 120, and 240 degrees.
%
%   AC = (sqrt(2)/3) * sqrt((I1-I2).^2 + (I2-I3).^2 + (I3-I1).^2)
%   DC = (I1 + I2 + I3) / 3
%
%   Reference: Cuccia et al., "Modulated imaging: quantitative analysis
%   and tomography of turbid media in the spatial-frequency domain,"
%   Opt. Lett. 30(11), 1354-1356 (2005).
%
%   This is a MATLAB port of src/sfdi_demodulation.py in this repository,
%   kept in sync so either language can be used interchangeably.

    if ~isequal(size(I1), size(I2), size(I3))
        error('sfdiDemodulate:sizeMismatch', 'I1, I2, I3 must have identical size');
    end

    I1 = double(I1); I2 = double(I2); I3 = double(I3);

    AC = (sqrt(2) / 3) * sqrt((I1 - I2).^2 + (I2 - I3).^2 + (I3 - I1).^2);
    DC = (I1 + I2 + I3) / 3;
end
