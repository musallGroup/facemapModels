function export_dennis_performance_csvs()

% ============================================================
% EXPORT PERFORMANCE CSVS FOR DENNIS COHORT
% Mice: 458, 460, 461, 462
% ============================================================

mice = ["458", "460", "461", "462"];

baseBhvPath = '\\Naskampa\lts\BpodBehavior';

outDir = '\\Naskampa\lts\Team\Mick\FM_Dennis_Cohort\Cohort_Summary\Performance';

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Same coding as previous Mick mice:
% 1 = Vision
% 4 = Tactile
% 5 = Multi
modIdxWanted = [1 4 5];

for mouseIdx = 1:numel(mice)

    mouseID = mice(mouseIdx);

    bhvPath = fullfile(baseBhvPath, char(mouseID), 'PuffyPenguin', 'Session Data');
    outCsv  = fullfile(outDir, "performance_" + mouseID + ".csv");

    fprintf('\n==============================\n');
    fprintf('Mouse %s\n', mouseID);
    fprintf('Behavior path: %s\n', bhvPath);

    if ~exist(bhvPath, 'dir')
        warning('Path not found: %s', bhvPath);
        continue
    end

    recs = dir(bhvPath);
    recs = recs([recs.isdir]);
    recs = recs(~ismember({recs.name},{'.','..'}));

    session_date = {};
    perfVision   = [];
    perfTactile  = [];
    perfMulti    = [];
    nVision      = [];
    nTactile     = [];
    nMulti       = [];

    cnt = 0;

    for iRec = 1:numel(recs)

        cRecFolder = fullfile(bhvPath, recs(iRec).name);

        cFile = dir(fullfile(cRecFolder, ['*' recs(iRec).name '.mat']));

        if isempty(cFile)
            continue
        end

        S = load(fullfile(cRecFolder, cFile(1).name));

        if ~isfield(S, 'SessionData')
            continue
        end

        bhv = S.SessionData;

        if ~isfield(bhv, 'Assisted') || sum(bhv.Assisted) <= 25
            continue
        end

        cnt = cnt + 1;

        dateStr = regexp(recs(iRec).name, '\d{8}', 'match', 'once');

        if isempty(dateStr)
            dateStr = regexp(cFile(1).name, '\d{8}', 'match', 'once');
        end

        if isempty(dateStr)
            dateStr = sprintf('unknown_%03d', cnt);
        end

        session_date{cnt,1} = dateStr;

        perfVals = nan(1,3);
        nVals    = zeros(1,3);

        for m = 1:numel(modIdxWanted)

            modCode = modIdxWanted(m);

            idxUse = bhv.StimType == modCode & ...
                     bhv.Assisted & ...
                     ~bhv.DidNotChoose & ...
                     bhv.Modality == 1;

            nVals(m) = sum(idxUse);

            if sum(idxUse) > 0
                perfVals(m) = sum(bhv.Rewarded(idxUse)) / sum(idxUse);
            end
        end

        perfVision(cnt,1)  = perfVals(1);
        perfTactile(cnt,1) = perfVals(2);
        perfMulti(cnt,1)   = perfVals(3);

        nVision(cnt,1)     = nVals(1);
        nTactile(cnt,1)    = nVals(2);
        nMulti(cnt,1)      = nVals(3);
    end

    if cnt == 0
        warning('No valid sessions found for mouse %s', mouseID);
        continue
    end

    T = table( ...
        string(session_date), ...
        perfVision, perfTactile, perfMulti, ...
        nVision, nTactile, nMulti, ...
        'VariableNames', { ...
            'session_date', ...
            'perfVision', 'perfTactile', 'perfMulti', ...
            'nVision', 'nTactile', 'nMulti'});

    T = sortrows(T, 'session_date');

    writetable(T, outCsv);

    fprintf('Saved: %s\n', outCsv);
    fprintf('Sessions exported: %d\n', height(T));
end

disp('Done.')

end