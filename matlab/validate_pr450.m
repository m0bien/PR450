% Validate the bit-exact parts of the pr450 LTE / NR generators against
% LTE Toolbox and 5G Toolbox. Run scripts/export_validation_vectors.py first.
%
%   matlab -batch "run('matlab/validate_pr450.m')"

if isempty(which('nrPRBS')) || isempty(which('lteCellRS'))
    error('validate_pr450 needs both LTE Toolbox and 5G Toolbox.');
end

here = fileparts(mfilename('fullpath'));
v = load(fullfile(here, 'pr450_vectors.mat'));
tol = 1e-9;
failures = 0;

% --- Gold sequence ---------------------------------------------------------
for i = 1:numel(v.gold_c_inits)
    ref = double(nrPRBS(v.gold_c_inits(i), 600)).';
    err = max(abs(ref - v.gold_bits(i, :)));
    failures = failures + report(sprintf('Gold c_init=%d', v.gold_c_inits(i)), err, tol);
end

for cell = v.cell_ids(:).'
    % --- LTE CRS / PSS / SSS over one frame -------------------------------
    for ports = [1 2]
        crs = complex(zeros(300, 140));
        sync = complex(zeros(300, 140));
        for sf = 0:9
            enb = struct('NDLRB', 25, 'NCellID', cell, 'CellRefP', ports, ...
                'CyclicPrefix', 'Normal', 'DuplexMode', 'FDD', 'NSubframe', sf, 'NFrame', 0);
            g = lteDLResourceGrid(enb);
            g(lteCellRSIndices(enb)) = lteCellRS(enb);
            cols = sf*14 + (1:14);
            crs(:, cols) = sum(g, 3);
            s = lteDLResourceGrid(enb);
            s(ltePSSIndices(enb)) = ltePSS(enb);
            s(lteSSSIndices(enb)) = lteSSS(enb);
            sync(:, cols) = sum(s, 3);
        end
        failures = failures + report(sprintf('LTE CRS  cell=%d ports=%d', cell, ports), ...
            max(abs(crs(:) - reshape(v.(sprintf('lte_crs_c%d_p%d', cell, ports)), [], 1))), tol);
        failures = failures + report(sprintf('LTE sync cell=%d ports=%d', cell, ports), ...
            max(abs(sync(:) - reshape(v.(sprintf('lte_sync_c%d_p%d', cell, ports)), [], 1))), tol);
    end

    % --- NR SSB: PSS / SSS / PBCH DMRS --------------------------------------
    for ibar = 0:3
        ssb = complex(zeros(240, 4));
        ssb(nrPSSIndices) = nrPSS(cell);
        ssb(nrSSSIndices) = nrSSS(cell);
        ssb(nrPBCHDMRSIndices(cell)) = nrPBCHDMRS(cell, ibar);
        mine = v.(sprintf('nr_ssb_c%d_i%d', cell, ibar));
        failures = failures + report(sprintf('NR SSB   cell=%d ibar=%d', cell, ibar), ...
            max(abs(ssb(:) - mine(:))), tol);
    end
end

if failures == 0
    fprintf('\nALL CHECKS PASSED\n');
else
    error('%d check(s) failed', failures);
end

function failed = report(name, err, tol)
    failed = err > tol;
    status = 'ok  ';
    if failed
        status = 'FAIL';
    end
    fprintf('%s  %-34s max|err| = %.3g\n', status, name, err);
end
