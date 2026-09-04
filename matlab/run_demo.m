%RUN_DEMO Small end-to-end MATLAB demo mirroring scripts/run_demo.py.
%   Generates a synthetic two-optical-property test image, forward
%   simulates R_d at two spatial frequencies, and inverts it back with
%   invertOpticalProperties, reporting the recovery RMSE.

rng(0);

muaTrue  = [0.01 0.02; 0.04 0.06];
muspTrue = [1.0 1.5; 2.0 2.5];

fxList = [0.0, 0.2];
RdMeasured = cell(1, numel(fxList));
for k = 1:numel(fxList)
    RdMeasured{k} = diffuseReflectance(muaTrue, muspTrue, fxList(k));
end

muaGrid = linspace(0.005, 0.08, 60);
muspGrid = linspace(0.4, 3.0, 60);
[muaRecovered, muspRecovered] = invertOpticalProperties(RdMeasured, fxList, muaGrid, muspGrid);

muaRmse = sqrt(mean((muaRecovered(:) - muaTrue(:)).^2));
muspRmse = sqrt(mean((muspRecovered(:) - muspTrue(:)).^2));

fprintf('mu_a  true:\n'); disp(muaTrue);
fprintf('mu_a  recovered:\n'); disp(muaRecovered);
fprintf('mu_a RMSE: %.5f mm^-1\n', muaRmse);
fprintf('mu_s'' RMSE: %.5f mm^-1\n', muspRmse);
