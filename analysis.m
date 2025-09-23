%% ========================================================================
%  MATLAB Analysis Automation for Pursuit-Evasion Training Logs
%  Generates mission diagnostics directly from Python training outputs.
% ========================================================================

if exist('analysis_config_override', 'var')
    temp_override = analysis_config_override; %#ok<NASGU>
    clearvars -except temp_override;
    analysis_config_override = temp_override; %#ok<NASGU>
    clear temp_override;
else
    clearvars;
end
close all; clc;
format long g;

% Global plotting defaults
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontSize', 12);
set(0, 'DefaultAxesGridAlpha', 0.3);
set(0, 'DefaultAxesXGrid', 'on');
set(0, 'DefaultAxesYGrid', 'on');
set(0, 'DefaultAxesZGrid', 'on');

%% ------------------------------------------------------------------------
%  CONFIGURATION
%% ------------------------------------------------------------------------
path_ctx = resolve_analysis_paths();

config.logs_root = path_ctx.logs_root;       % Root directory for training logs
config.target_run = '';            % Specify run folder or leave empty for latest
config.experiment_filter = '*';    % Wildcard filter (e.g., 'interactive_training*')
config.save_figures = false;       % Save figures automatically
config.figure_formats = {'png'};   % Formats used when saving figures
config.output_dir = '';            % Optional override for figure output path
config.verbose = true;             % Display progress information

% Additional switches for test-result batches
config.analysis_mode = 'training';     % 'training' or 'test'
config.test_results_root = path_ctx.test_results_root;
config.target_test_run = '';
config.test_case_filter = [];          % e.g., [1 3 5] to inspect selected tests

if exist('analysis_config_override', 'var') && isstruct(analysis_config_override)
    override_struct = analysis_config_override;
    clear analysis_config_override;
    config = apply_config_override(config, override_struct);
    if isfield(override_struct, 'logs_root')
        path_ctx.logs_root = override_struct.logs_root;
    end
    if isfield(override_struct, 'test_results_root')
        path_ctx.test_results_root = override_struct.test_results_root;
    end
end

%% ------------------------------------------------------------------------
%  DATA SELECTION AND PREPARATION
%% ------------------------------------------------------------------------
analysis_cases = struct([]);
export_root = '';

mode = lower(char(string(config.analysis_mode)));

switch mode
    case 'training'
        [run_info, config] = resolve_run_directory(config);
        plots_dir = fullfile(run_info.path, 'plots');
        analysis_dir = fullfile(run_info.path, 'analysis');
        run_label = derive_run_label(run_info);
        if isempty(config.output_dir)
            export_root = fullfile(config.logs_root, 'matlab_analysis_training', run_label);
        else
            export_root = config.output_dir;
        end
        if config.save_figures && ~exist(export_root, 'dir')
            mkdir(export_root);
        end

        if config.verbose
            fprintf('=====================================\n');
            fprintf('Selected training run : %s\n', run_info.name);
            fprintf('Log directory         : %s\n', run_info.path);
            fprintf('Output directory      : %s\n', export_root);
            fprintf('=====================================\n\n');
        end

        files.stats = fullfile(plots_dir, 'evasion_stats.json');
        files.progress = fullfile(plots_dir, 'training_progress.json');
        files.delta_v_csv = fullfile(plots_dir, 'evader_delta_v.csv');
        files.analysis_json = fullfile(analysis_dir, 'analysis_data.json');
        files.ephemeris_mat = fullfile(run_info.path, 'final_episode_ephemeris.mat');

        stats = load_json_if_exists(files.stats);
        progress = load_json_if_exists(files.progress);
        analysis_data = load_json_if_exists(files.analysis_json);
        delta_v_table = load_table_if_exists(files.delta_v_csv);
        ephemeris = load_mat_if_exists(files.ephemeris_mat);

        episodes = extract_episode_array(analysis_data);
        primary_episode = struct();
        if ~isempty(episodes)
            primary_episode = episodes(1);
        end

        trajectory = struct();
        if ~isempty(primary_episode) && isfield(primary_episode, 'positions')
            pos_mat = ensure_matrix(primary_episode.positions, 3);
            vel_mat = ensure_matrix(primary_episode.velocities, 3);
            trajectory.x = pos_mat(:, 1);
            trajectory.y = pos_mat(:, 2);
            trajectory.z = pos_mat(:, 3);
            trajectory.vx = vel_mat(:, 1);
            trajectory.vy = vel_mat(:, 2);
            trajectory.vz = vel_mat(:, 3);
            trajectory.distances = ensure_vector(primary_episode.distances);
            trajectory.times = ensure_vector(primary_episode.times);
        end

        has_xyz = isfield(trajectory, 'x') && isfield(trajectory, 'y') && isfield(trajectory, 'z');
        if has_xyz && (~isfield(trajectory, 'distances') || isempty(trajectory.distances))
            trajectory.distances = sqrt(trajectory.x.^2 + trajectory.y.^2 + trajectory.z.^2);
        end

        if ~isfield(trajectory, 'times') || isempty(trajectory.times)
            if isfield(trajectory, 'distances') && ~isempty(trajectory.distances)
                n_points = numel(trajectory.distances);
            elseif has_xyz
                n_points = numel(trajectory.x);
            else
                n_points = 0;
            end
            if n_points > 0
                trajectory.times = (0:n_points-1)';
            end
        end

        step_dv = struct();
        if ~isempty(primary_episode)
            step_dv.evader_vectors = ensure_matrix(primary_episode.evader_dv_vectors, 3);
            step_dv.pursuer_vectors = ensure_matrix(primary_episode.pursuer_dv_vectors, 3);
            step_dv.evader_magnitude = ensure_vector(primary_episode.evader_dvs);
            step_dv.pursuer_magnitude = ensure_vector(primary_episode.pursuer_dvs);
            step_dv.times = trajectory.times;
            if isempty(step_dv.evader_magnitude) && ~isempty(step_dv.evader_vectors)
                step_dv.evader_magnitude = vecnorm(step_dv.evader_vectors, 2, 2);
            end
            if isempty(step_dv.pursuer_magnitude) && ~isempty(step_dv.pursuer_vectors)
                step_dv.pursuer_magnitude = vecnorm(step_dv.pursuer_vectors, 2, 2);
            end
        end

        per_episode_dv = [];
        if isstruct(stats) && isfield(stats, 'evader_delta_v') && ~isempty(stats.evader_delta_v)
            per_episode_dv = stats.evader_delta_v(:);
        end
        if isempty(per_episode_dv) && istable(delta_v_table)
            if ismember('delta_v', delta_v_table.Properties.VariableNames)
                per_episode_dv = delta_v_table.delta_v(:);
            elseif ismember('Var2', delta_v_table.Properties.VariableNames)
                per_episode_dv = delta_v_table.Var2(:);
            end
        end

        mission_outcome = 'UNKNOWN';
        if ~isempty(primary_episode) && isfield(primary_episode, 'outcome')
            mission_outcome = upper(string(primary_episode.outcome));
        elseif isstruct(stats) && isfield(stats, 'episodes_info') && ~isempty(stats.episodes_info)
            mission_outcome = upper(string(stats.episodes_info(end).outcome));
        end

        case_struct = struct( ...
            'label', char(run_info.name), ...
            'file_prefix', sanitize_label(run_info.name), ...
            'trajectory', trajectory, ...
            'step_dv', step_dv, ...
            'ephemeris', ephemeris, ...
            'per_episode_dv', ensure_vector(per_episode_dv), ...
            'progress', progress, ...
            'stats', stats, ...
            'mission_outcome', mission_outcome, ...
            'export_dir', export_root);

        analysis_cases = case_struct;

    case 'test'
        test_config = struct( ...
            'logs_root', config.test_results_root, ...
            'target_run', config.target_test_run, ...
            'experiment_filter', '*');

        [run_info, ~] = resolve_run_directory(test_config);

        run_label = derive_run_label(run_info);
        if isempty(config.output_dir)
            export_root = fullfile(config.test_results_root, 'matlab_analysis_evaluation', run_label);
        else
            export_root = config.output_dir;
        end
        if config.save_figures && ~exist(export_root, 'dir')
            mkdir(export_root);
        end

        if config.verbose
            fprintf('=====================================\n');
            fprintf('Selected test batch     : %s\n', run_info.name);
            fprintf('Test results directory  : %s\n', run_info.path);
            fprintf('Output directory        : %s\n', export_root);
            fprintf('=====================================\n\n');
        end

        summary_csv = fullfile(run_info.path, 'test_results.csv');
        summary_entries = normalize_test_summary(load_table_if_exists(summary_csv));

        traj_files = dir(fullfile(run_info.path, 'test_*_trajectory_trajectory_data.json'));
        test_ids = arrayfun(@(f) parse_test_identifier(f.name), traj_files);
        test_ids = unique(test_ids(~isnan(test_ids)));

        if ~isempty(config.test_case_filter)
            filter_ids = config.test_case_filter(:)';
            test_ids = intersect(test_ids, filter_ids);
        end
        test_ids = sort(test_ids);

        if isempty(test_ids)
            error('No test trajectory files found in %s', run_info.path);
        end

        analysis_cases = struct([]);
        for idx = 1:numel(test_ids)
            test_id = test_ids(idx);
            label = sprintf('test_%d', test_id);
            file_prefix = sanitize_label(label);

            trajectory = struct();
            traj_json = load_json_if_exists(fullfile(run_info.path, sprintf('%s_trajectory_trajectory_data.json', label)));
            if isstruct(traj_json) && ~isempty(fieldnames(traj_json))
                components = {'x', 'y', 'z', 'vx', 'vy', 'vz'};
                for cidx = 1:numel(components)
                    field = components{cidx};
                    if isfield(traj_json, field)
                        trajectory.(field) = ensure_vector(traj_json.(field));
                    end
                end
            end

            if isfield(trajectory, 'x') && ~isempty(trajectory.x) && ...
                    isfield(trajectory, 'y') && ~isempty(trajectory.y) && ...
                    isfield(trajectory, 'z') && ~isempty(trajectory.z)
                trajectory.distances = sqrt(trajectory.x.^2 + trajectory.y.^2 + trajectory.z.^2);
            end

            if ~isfield(trajectory, 'times') || isempty(trajectory.times)
                trajectory.times = [];
            end

            step_dv = struct();
            dv_json = load_json_if_exists(fullfile(run_info.path, sprintf('%s_delta_v.json', label)));
            if isstruct(dv_json) && ~isempty(fieldnames(dv_json))
                if isfield(dv_json, 'evader_delta_v')
                    step_dv.evader_vectors = ensure_matrix(dv_json.evader_delta_v, 3);
                end
                if isfield(dv_json, 'pursuer_delta_v')
                    step_dv.pursuer_vectors = ensure_matrix(dv_json.pursuer_delta_v, 3);
                end
                if isfield(dv_json, 'evader_delta_v_norm')
                    step_dv.evader_magnitude = ensure_vector(dv_json.evader_delta_v_norm);
                end
                if isfield(dv_json, 'pursuer_delta_v_norm')
                    step_dv.pursuer_magnitude = ensure_vector(dv_json.pursuer_delta_v_norm);
                end
                if isfield(dv_json, 'time')
                    step_dv.times = ensure_vector(dv_json.time);
                end
                if (~isfield(step_dv, 'evader_magnitude') || isempty(step_dv.evader_magnitude)) && isfield(step_dv, 'evader_vectors')
                    step_dv.evader_magnitude = vecnorm(step_dv.evader_vectors, 2, 2);
                end
                if (~isfield(step_dv, 'pursuer_magnitude') || isempty(step_dv.pursuer_magnitude)) && isfield(step_dv, 'pursuer_vectors')
                    step_dv.pursuer_magnitude = vecnorm(step_dv.pursuer_vectors, 2, 2);
                end
            end

            if (~isfield(trajectory, 'times') || isempty(trajectory.times)) && isfield(step_dv, 'times') && ~isempty(step_dv.times)
                trajectory.times = step_dv.times;
            end
            if (~isfield(trajectory, 'distances') || isempty(trajectory.distances)) && isfield(trajectory, 'x') && ~isempty(trajectory.x)
                trajectory.distances = sqrt(trajectory.x.^2 + trajectory.y.^2 + trajectory.z.^2);
            end
            if (~isfield(trajectory, 'times') || isempty(trajectory.times)) && isfield(trajectory, 'x') && ~isempty(trajectory.x)
                trajectory.times = (0:numel(trajectory.x)-1)';
            end

            ephemeris = struct();
            eci_json = load_json_if_exists(fullfile(run_info.path, sprintf('%s_eci_eci.json', label)));
            if isstruct(eci_json) && isfield(eci_json, 't') && ~isempty(eci_json.t)
                ephemeris.t = ensure_vector(eci_json.t);
                evader_x = ensure_vector(extract_field(eci_json, {'evader_x'}));
                evader_y = ensure_vector(extract_field(eci_json, {'evader_y'}));
                evader_z = ensure_vector(extract_field(eci_json, {'evader_z'}));
                pursuer_x = ensure_vector(extract_field(eci_json, {'pursuer_x'}));
                pursuer_y = ensure_vector(extract_field(eci_json, {'pursuer_y'}));
                pursuer_z = ensure_vector(extract_field(eci_json, {'pursuer_z'}));
                if ~isempty(evader_x) && ~isempty(evader_y) && ~isempty(evader_z)
                    ephemeris.evader = [evader_x, evader_y, evader_z];
                end
                if ~isempty(pursuer_x) && ~isempty(pursuer_y) && ~isempty(pursuer_z)
                    ephemeris.pursuer = [pursuer_x, pursuer_y, pursuer_z];
                end
            end

            summary = find_summary_for_test(summary_entries, test_id);
            success_flag = NaN;
            per_episode_value = [];
            if ~isempty(fieldnames(summary))
                if isfield(summary, 'success')
                    success_flag = parse_boolean(summary.success);
                end
                per_episode_value = extract_field(summary, { ...
                    'evader_total_delta_v_ms', ...
                    'total_evader_delta_v', ...
                    'evader_delta_v'});
                if isempty(per_episode_value)
                    per_episode_value = extract_field(summary, {'total_evader_delta_v_ms'});
                end
            end

            if isempty(per_episode_value) && isfield(step_dv, 'evader_magnitude') && ~isempty(step_dv.evader_magnitude)
                per_episode_value = step_dv.evader_magnitude(end);
            end

            if isnan(success_flag)
                mission_outcome = 'UNKNOWN';
            elseif success_flag
                mission_outcome = 'SUCCESS';
            else
                mission_outcome = 'FAILURE';
            end

            per_episode_vec = ensure_vector(per_episode_value);
            progress_case = struct();
            if ~isnan(success_flag)
                progress_case.success_rates = double(success_flag);
            else
                progress_case.success_rates = [];
            end

            stats_case = struct();
            if ~isnan(success_flag)
                stats_case.total_episodes = 1;
                stats_case.final_success_rate = double(success_flag);
                stats_case.captures = double(~success_flag);
                stats_case.fuel_depleted = 0;
                stats_case.episodes_info = struct('outcome', mission_outcome);
            else
                stats_case.total_episodes = [];
                stats_case.final_success_rate = [];
                stats_case.captures = [];
                stats_case.fuel_depleted = [];
                stats_case.episodes_info = [];
            end

            case_struct = struct( ...
                'label', label, ...
                'file_prefix', file_prefix, ...
                'trajectory', trajectory, ...
                'step_dv', step_dv, ...
                'ephemeris', ephemeris, ...
                'per_episode_dv', per_episode_vec, ...
                'progress', progress_case, ...
                'stats', stats_case, ...
                'mission_outcome', mission_outcome, ...
                'export_dir', export_root);

            if isempty(analysis_cases)
                analysis_cases = case_struct;
            else
                analysis_cases(end + 1) = case_struct; %#ok<AGROW>
            end
        end

    otherwise
        error('Unsupported analysis_mode: %s', config.analysis_mode);
end

%% ------------------------------------------------------------------------
%  GLOBAL CONSTANTS USED DURING VISUALIZATION
%% ------------------------------------------------------------------------
const.R_EARTH = 6378e3;                % Earth radius [m]
const.MU_EARTH = 3.986004418e14;       % Earth gravitational parameter [m^3/s^2]
const.CAPTURE_RADIUS = 1e3;            % Capture distance [m]
const.EVASION_RADIUS = 5e3;            % Evasion distance [m]

if isempty(analysis_cases)
    error('No analysis cases prepared for visualization.');
end

num_cases = numel(analysis_cases);
for idx = 1:num_cases
    case_data = analysis_cases(idx);
    if config.verbose
        fprintf('Processing case %d/%d : %s\n', idx, num_cases, case_data.label);
    end
    generate_case_figures(case_data, const, config);
end

if config.verbose
    fprintf('\nAnalysis complete (%d case(s)).\n', num_cases);
    if config.save_figures
        fprintf('Figures saved to: %s\n', export_root);
    end
end

%% ========================================================================
%  HELPER FUNCTIONS
% ========================================================================
function config = apply_config_override(config, override)
    fields = fieldnames(override);
    for idx = 1:numel(fields)
        field = fields{idx};
        config.(field) = override.(field);
    end
end

function ctx = resolve_analysis_paths()
    script_dir = get_script_directory();
    project_root = find_project_root(script_dir);

    logs_root = fullfile(project_root, 'logs');
    test_root = fullfile(project_root, 'test_results');

    ctx = struct( ...
        'script_dir', script_dir, ...
        'project_root', project_root, ...
        'logs_root', logs_root, ...
        'test_results_root', test_root);
end

function script_dir = get_script_directory()
    script_dir = '';

    stack = dbstack('-completenames');
    for idx = numel(stack):-1:1
        candidate = stack(idx).file;
        if ~isempty(candidate) && exist(candidate, 'file')
            script_dir = fileparts(candidate);
            if ~isempty(script_dir)
                return;
            end
        end
    end

    mf = mfilename('fullpath');
    if ~isempty(mf)
        script_dir = fileparts(mf);
    end

    if isempty(script_dir)
        script_dir = pwd;
    end
end

function project_root = find_project_root(start_dir)
    if nargin == 0 || isempty(start_dir)
        start_dir = pwd;
    end

    current = start_dir;
    last_valid = start_dir;

    while true
        logs_exists = exist(fullfile(current, 'logs'), 'dir');
        tests_exists = exist(fullfile(current, 'test_results'), 'dir');
        if logs_exists || tests_exists
            project_root = current;
            return;
        end

        parent = fileparts(current);
        if isempty(parent) || strcmp(parent, current)
            break;
        end
        last_valid = current;
        current = parent;
    end

    project_root = last_valid;
end

function generate_case_figures(case_data, const, config)
    trajectory = case_data.trajectory;
    step_dv = case_data.step_dv;
    ephemeris = case_data.ephemeris;
    per_episode_dv = case_data.per_episode_dv;
    progress = case_data.progress;
    stats = case_data.stats;

    if ~isstruct(progress)
        progress = struct();
    end
    if ~isstruct(stats)
        stats = struct();
    end

    mission_outcome_str = upper(char(string(case_data.mission_outcome)));
    file_prefix = sanitize_label(case_data.file_prefix);
    export_dir = case_data.export_dir;

    traj_x = [];
    traj_y = [];
    traj_z = [];
    traj_vx = [];
    traj_vy = [];
    traj_vz = [];

    has_trajectory = isstruct(trajectory) && all(isfield(trajectory, {'x', 'y', 'z'})) && ...
        ~isempty(trajectory.x);

    rel_distance = [];
    rel_velocity = [];
    time_axis = [];

    if has_trajectory
        traj_x = ensure_vector(trajectory.x);
        traj_y = ensure_vector(get_field_or_default(trajectory, 'y', zeros(size(traj_x))));
        traj_z = ensure_vector(get_field_or_default(trajectory, 'z', zeros(size(traj_x))));
        time_axis = ensure_vector(get_field_or_default(trajectory, 'times', (0:numel(traj_x)-1)'));
        traj_vx = ensure_vector(get_field_or_default(trajectory, 'vx', zeros(size(traj_x))));
        traj_vy = ensure_vector(get_field_or_default(trajectory, 'vy', zeros(size(traj_x))));
        traj_vz = ensure_vector(get_field_or_default(trajectory, 'vz', zeros(size(traj_x))));
        rel_distance = ensure_vector(get_field_or_default(trajectory, 'distances', sqrt(traj_x.^2 + traj_y.^2 + traj_z.^2)));
        rel_velocity = sqrt(traj_vx.^2 + traj_vy.^2 + traj_vz.^2);
    end

    dt_seconds = [];
    if numel(time_axis) > 1
        dt_seconds = median(diff(time_axis));
    end

    has_deltav = isstruct(step_dv) && isfield(step_dv, 'evader_magnitude') && ~isempty(step_dv.evader_magnitude);
    if has_deltav
        evader_cumulative = cumsum(step_dv.evader_magnitude);
        if isfield(step_dv, 'pursuer_magnitude') && ~isempty(step_dv.pursuer_magnitude)
            pursuer_cumulative = cumsum(step_dv.pursuer_magnitude);
        else
            pursuer_cumulative = [];
        end
    else
        evader_cumulative = [];
        pursuer_cumulative = [];
    end

    has_ephemeris = isstruct(ephemeris) && ~isempty(ephemeris) && ...
        all(isfield(ephemeris, {'t', 'evader', 'pursuer'}));
    if has_ephemeris
        has_ephemeris = ~isempty(ephemeris.evader) && ~isempty(ephemeris.pursuer);
    end

    if ~isempty(per_episode_dv)
        per_episode_dv = per_episode_dv(~isnan(per_episode_dv));
    end

    t = time_axis;
    if isempty(t)
        t = (0:numel(rel_distance)-1)';
    end

    fig1 = figure('Name', sprintf('%s - Relative Trajectory (LVLH)', case_data.label), ...
        'Position', [80, 80, 1400, 900], 'Color', 'white');

    if has_trajectory
        subplot(2, 3, [1, 4]);
        hold on; grid on; box on;

        traj_x = traj_x(:);
        traj_y = traj_y(:);
        traj_z = traj_z(:);
        coords = [traj_x, traj_y, traj_z];

        trajectory_color = [0.0, 0.4470, 0.7410];
        evader_color = [0.0, 0.6, 0.0];
        pursuer_color = [0.8, 0.2, 0.2];

        traj_handle = plot3(traj_x, traj_y, traj_z, 'Color', trajectory_color, 'LineWidth', 2.2, ...
            'DisplayName', 'Relative Trajectory');
        hold on;
        start_handle = plot3(traj_x(1), traj_y(1), traj_z(1), '^', 'MarkerSize', 9, ...
            'MarkerFaceColor', evader_color, 'MarkerEdgeColor', 'k', 'LineStyle', 'none', ...
            'DisplayName', 'Start');
        end_handle = plot3(traj_x(end), traj_y(end), traj_z(end), 'x', 'MarkerSize', 9, ...
            'MarkerEdgeColor', pursuer_color, 'LineWidth', 1.8, 'LineStyle', 'none', ...
            'DisplayName', 'End');
        evader_handle = plot3(0, 0, 0, 'p', 'MarkerSize', 11, 'MarkerFaceColor', [0 0 0], ...
            'MarkerEdgeColor', 'k', 'LineStyle', 'none', 'DisplayName', 'Evader (Origin)');

        draw_zone_spheres(const);

        range_x = max(traj_x, [], 'omitnan') - min(traj_x, [], 'omitnan');
        range_y = max(traj_y, [], 'omitnan') - min(traj_y, [], 'omitnan');
        range_z = max(traj_z, [], 'omitnan') - min(traj_z, [], 'omitnan');
        ranges = [range_x, range_y, range_z];
        max_range = max(ranges);
        if ~isfinite(max_range) || max_range <= 0
            max_range = max(vecnorm(coords, 2, 2), [], 'omitnan');
        end
        if isempty(max_range) || max_range <= 0
            max_range = 1.0;
        end

        arrow_length = 0.05 * max_range;
        if size(coords, 1) > 1
            step_norms = vecnorm(diff(coords, 1, 1), 2, 2);
            step_norms = step_norms(isfinite(step_norms) & step_norms > 0);
            if ~isempty(step_norms)
                sorted_steps = sort(step_norms);
                idx_percentile = max(1, round(0.75 * numel(sorted_steps)));
                typical_step = sorted_steps(idx_percentile);
                arrow_length = min(arrow_length, typical_step * 0.8);
            end
        end
        if arrow_length <= 0
            arrow_length = max_range * 0.1;
        end

        pursuer_handle = [];
        if isfield(step_dv, 'pursuer_vectors') && ~isempty(step_dv.pursuer_vectors)
            dv_p = step_dv.pursuer_vectors;
            n_p = size(dv_p, 1);
            n_traj = numel(traj_x);
            max_count = min(n_p, n_traj);
            dv_p = dv_p(1:max_count, :);
            traj_subset = coords(1:max_count, :);
            magnitudes = vecnorm(dv_p, 2, 2);
            valid_idx = find(isfinite(magnitudes) & magnitudes > 0);
            for vidx = valid_idx(:)'
                direction = dv_p(vidx, :);
                scale_factor = arrow_length / magnitudes(vidx);
                scaled_vec = direction * scale_factor;
                base_pos = traj_subset(vidx, :);
                q = quiver3(base_pos(1), base_pos(2), base_pos(3), ...
                    scaled_vec(1), scaled_vec(2), scaled_vec(3), 0);
                q.Color = pursuer_color;
                q.LineWidth = 1.1;
                q.MaxHeadSize = 0.6;
                if isempty(pursuer_handle)
                    q.DisplayName = 'Pursuer ΔV';
                    pursuer_handle = q;
                else
                    q.HandleVisibility = 'off';
                end
            end
        end

        evader_impulse_handle = [];
        if isfield(step_dv, 'evader_vectors') && ~isempty(step_dv.evader_vectors)
            dv_e = step_dv.evader_vectors;
            magnitudes = vecnorm(dv_e, 2, 2);
            valid_idx = find(isfinite(magnitudes) & magnitudes > 0);
            for vidx = valid_idx(:)'
                direction = dv_e(vidx, :);
                scale_factor = arrow_length / magnitudes(vidx);
                scaled_vec = direction * scale_factor;
                q = quiver3(0, 0, 0, scaled_vec(1), scaled_vec(2), scaled_vec(3), 0);
                q.Color = evader_color;
                q.LineWidth = 1.0;
                q.MaxHeadSize = 0.6;
                if isempty(evader_impulse_handle)
                    q.DisplayName = 'Evader ΔV';
                    evader_impulse_handle = q;
                else
                    q.HandleVisibility = 'off';
                end
            end
        end

        xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
        title(sprintf('%s - Outcome: %s', case_data.label, mission_outcome_str));
        axis equal; view(45, 25);

        legend_handles = [traj_handle, start_handle, end_handle, evader_handle];
        if ~isempty(evader_impulse_handle)
            legend_handles(end+1) = evader_impulse_handle; %#ok<AGROW>
        end
        if ~isempty(pursuer_handle)
            legend_handles(end+1) = pursuer_handle; %#ok<AGROW>
        end
        legend(legend_handles, 'Location', 'best');
    else
        subplot(2, 3, [1, 4]);
        axis off;
        text(0.5, 0.5, 'Relative position data not available', 'HorizontalAlignment', 'center');
    end

    subplot(2, 3, 2);
    if ~isempty(rel_distance)
        hold on; grid on; box on;
        plot(t, rel_distance, 'b-', 'LineWidth', 2);
        yline(const.CAPTURE_RADIUS, 'r--', 'LineWidth', 1.2);
        yline(const.EVASION_RADIUS, 'g--', 'LineWidth', 1.2);
        xlabel('Time [s]'); ylabel('Distance [m]');
        title('Relative Distance Evolution');
    else
        axis off; text(0.5, 0.5, 'Distance series unavailable', 'HorizontalAlignment', 'center');
    end

    subplot(2, 3, 3);
    if ~isempty(rel_velocity)
        hold on; grid on; box on;
        plot(t, traj_vx, 'r-');
        plot(t, traj_vy, 'g-');
        plot(t, traj_vz, 'b-');
        plot(t, rel_velocity, 'k--', 'LineWidth', 2);
        xlabel('Time [s]'); ylabel('Velocity [m/s]');
        title('Relative Velocity Components');
        legend('V_x', 'V_y', 'V_z', '|V|', 'Location', 'best');
    else
        axis off; text(0.5, 0.5, 'Velocity data not available', 'HorizontalAlignment', 'center');
    end

    subplot(2, 3, 5);
    if ~isempty(rel_distance)
        hold on; grid on; box on;
        scatter(traj_x, traj_vx, 20, linspace(0, 1, numel(traj_x)), 'filled');
        xlabel('Position X [m]'); ylabel('Velocity V_x [m/s]');
        title('Phase Space (X vs V_x)');
        cb_phase = colorbar;
        ylabel(cb_phase, 'Step progression');
    else
        axis off; text(0.5, 0.5, 'Phase space unavailable', 'HorizontalAlignment', 'center');
    end

    subplot(2, 3, 6);
    if ~isempty(rel_distance)
        hold on; grid on; box on;
        distance_rate = [0; diff(rel_distance)];
        plot(t, distance_rate, 'b-', 'LineWidth', 1.5);
        yline(0, 'k--');
        xlabel('Time [s]'); ylabel('Delta Distance [m]');
        title('Closing / Opening Rate');
    else
        axis off;
    end

    save_figure_if_needed(fig1, sprintf('%s_relative_trajectory', file_prefix), export_dir, config);

    fig2 = figure('Name', sprintf('%s - ECI Trajectory', case_data.label), ...
        'Position', [100, 100, 1400, 900], 'Color', 'white');
    if has_ephemeris
        plot_eci_diagnostics(ephemeris, const);
    else
        axis off;
        text(0.5, 0.5, 'ECI data not available (enable EphemerisLoggerCallback)', 'HorizontalAlignment', 'center');
    end
    save_figure_if_needed(fig2, sprintf('%s_eci_trajectory', file_prefix), export_dir, config);

    fig3 = figure('Name', sprintf('%s - Delta-V Analysis', case_data.label), ...
        'Position', [120, 120, 1400, 900], 'Color', 'white');
    if has_deltav
        plot_delta_v_diagnostics(step_dv, evader_cumulative, pursuer_cumulative, dt_seconds, t);
    else
        axis off;
        text(0.5, 0.5, 'Stepwise delta-v data not available', 'HorizontalAlignment', 'center');
    end
    save_figure_if_needed(fig3, sprintf('%s_delta_v_analysis', file_prefix), export_dir, config);

    fig4 = figure('Name', sprintf('%s - Mission Statistics', case_data.label), ...
        'Position', [140, 140, 1200, 800], 'Color', 'white');
    plot_mission_statistics(rel_distance, rel_velocity, evader_cumulative, pursuer_cumulative, per_episode_dv, progress, stats, const);
    save_figure_if_needed(fig4, sprintf('%s_mission_statistics', file_prefix), export_dir, config);
end

function label = sanitize_label(input)
    str = string(input);
    label = regexprep(str, '[^A-Za-z0-9_-]', '_');
    label = char(label);
end

function entries = normalize_test_summary(table_data)
    entries = struct([]);
    if ~istable(table_data)
        return;
    end
    entries = table2struct(table_data);
    for idx = 1:numel(entries)
        if isfield(entries(idx), 'test_run')
            entries(idx).test_run = coerce_to_double(entries(idx).test_run);
        end
        if isfield(entries(idx), 'success')
            entries(idx).success = parse_boolean(entries(idx).success);
        end
        numeric_fields = {'evader_delta_v', 'evader_total_delta_v', 'evader_total_delta_v_ms', ...
            'total_evader_delta_v', 'total_evader_delta_v_ms', 'final_distance', 'final_distance_m'};
        for fidx = 1:numel(numeric_fields)
            field = numeric_fields{fidx};
            if isfield(entries(idx), field)
                entries(idx).(field) = coerce_to_double(entries(idx).(field));
            end
        end
    end
end

function id = parse_test_identifier(filename)
    tokens = regexp(filename, '^test_(\d+)_', 'tokens', 'once');
    if isempty(tokens)
        id = NaN;
    else
        id = str2double(tokens{1});
    end
end

function summary = find_summary_for_test(entries, test_id)
    summary = struct();
    if isempty(entries)
        return;
    end
    target_idx = test_id - 1;
    for idx = 1:numel(entries)
        if ~isfield(entries(idx), 'test_run')
            continue;
        end
        run_idx = coerce_to_double(entries(idx).test_run);
        if isnan(run_idx)
            continue;
        end
        if run_idx == target_idx
            summary = entries(idx);
            return;
        end
    end
end

function value = extract_field(data, names)
    value = [];
    if ~isstruct(data)
        return;
    end
    names = cellstr(names);
    for idx = 1:numel(names)
        name = names{idx};
        if isfield(data, name) && ~isempty(data.(name))
            value = data.(name);
            return;
        end
    end
end

function tf = parse_boolean(value)
    if isempty(value)
        tf = false;
        return;
    end
    if islogical(value)
        tf = logical(value(1));
        return;
    end
    if isnumeric(value)
        tf = value(1) ~= 0;
        return;
    end
    str = lower(strtrim(string(value(1))));
    tf = any(str == "true" | str == "1" | str == "yes" | str == "success");
end

function val = coerce_to_double(value)
    if isempty(value)
        val = NaN;
        return;
    end
    if isnumeric(value)
        val = double(value(1));
        return;
    end
    str = string(value(1));
    val = str2double(str);
end

function out = get_field_or_default(data, name, default_value)
    if isstruct(data) && isfield(data, name) && ~isempty(data.(name))
        out = data.(name);
    else
        out = default_value;
    end
end

function label = derive_run_label(run_info)
    label = 'run';
    if ~isstruct(run_info)
        return;
    end
    if ~isfield(run_info, 'name') || isempty(run_info.name)
        return;
    end

    raw_name = string(run_info.name);
    raw_name = raw_name(1);
    [~, base_name, ~] = fileparts(raw_name);
    if strlength(base_name) == 0
        base_name = raw_name;
    end

    label = sanitize_label(base_name);
end

function [run_info, config] = resolve_run_directory(config)
    arguments
        config struct
    end

    listing = dir(fullfile(config.logs_root, config.experiment_filter));
    listing = listing([listing.isdir]);
    listing = listing(~ismember({listing.name}, {'.', '..'}));

    if isempty(listing)
        error('No log directories found in %s', config.logs_root);
    end

    if ~isempty(config.target_run)
        target_path = fullfile(config.logs_root, config.target_run);
        if ~exist(target_path, 'dir')
            error('Specified run directory not found: %s', target_path);
        end
        idx = find(strcmp({listing.name}, config.target_run), 1);
        if isempty(idx)
            warning('Specified run not in filtered list; using explicit path.');
            run_info.name = config.target_run;
            run_info.path = target_path;
            return;
        end
    else
        [~, idx] = max([listing.datenum]);
    end

    run_info.name = listing(idx).name;
    run_info.path = fullfile(listing(idx).folder, listing(idx).name);
end

function data = load_json_if_exists(filepath)
    if exist(filepath, 'file')
        raw = fileread(filepath);
        data = jsondecode(raw);
    else
        data = struct();
    end
end

function tbl = load_table_if_exists(filepath)
    if exist(filepath, 'file')
        opts = detectImportOptions(filepath, 'NumHeaderLines', 0);
        tbl = readtable(filepath, opts);
    else
        tbl = struct();
    end
end

function data = load_mat_if_exists(filepath)
    if exist(filepath, 'file')
        data = load(filepath);
    else
        data = struct();
    end
end

function episodes = extract_episode_array(analysis_data)
    episodes = [];
    if ~isstruct(analysis_data)
        return;
    end
    if isfield(analysis_data, 'all_episodes_data')
        episodes = analysis_data.all_episodes_data;
    end
    if iscell(episodes)
        episodes = [episodes{:}];
    end
end

function mat = ensure_matrix(value, expected_cols)
    if nargin < 2
        expected_cols = [];
    end
    if isempty(value)
        mat = [];
        return;
    end
    if iscell(value)
        rows = cellfun(@(row) reshape(double(row), 1, []), value, 'UniformOutput', false);
        mat = cell2mat(rows(:));
    elseif isnumeric(value)
        mat = double(value);
    else
        mat = [];
        return;
    end
    if ~isempty(expected_cols) && size(mat, 2) ~= expected_cols
        mat = reshape(mat, [], expected_cols);
    end
end

function vec = ensure_vector(value)
    if isempty(value)
        vec = [];
        return;
    end
    if iscell(value)
        vec = cellfun(@double, value);
    elseif isnumeric(value)
        vec = double(value);
    else
        vec = [];
    end
    vec = vec(:);
end

function draw_zone_spheres(const)
    [xs, ys, zs] = sphere(40);
    surf(xs * const.CAPTURE_RADIUS, ys * const.CAPTURE_RADIUS, zs * const.CAPTURE_RADIUS, ...
         'FaceColor', [0.8 0 0], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    surf(xs * const.EVASION_RADIUS, ys * const.EVASION_RADIUS, zs * const.EVASION_RADIUS, ...
         'FaceColor', [0 0.8 0], 'FaceAlpha', 0.10, 'EdgeColor', 'none');
end

function plot_eci_diagnostics(ephemeris, const)
    evader = ephemeris.evader;
    pursuer = ephemeris.pursuer;
    t = ephemeris.t;

    subplot(2, 3, [1, 4]);
    hold on; grid on; box on;
    [xe, ye, ze] = sphere(60);
    earth = surf(xe * const.R_EARTH, ye * const.R_EARTH, ze * const.R_EARTH);
    set(earth, 'FaceColor', [0.4 0.6 1], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    plot3(evader(:,1), evader(:,2), evader(:,3), 'Color', [0 0.8 0], 'LineWidth', 2);
    plot3(pursuer(:,1), pursuer(:,2), pursuer(:,3), 'Color', [0.8 0 0], 'LineWidth', 2);
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    title('ECI Trajectories'); axis equal; view(45, 25);

    subplot(2, 3, 2);
    hold on; grid on; box on;
    evader_radius = vecnorm(evader, 2, 2);
    pursuer_radius = vecnorm(pursuer, 2, 2);
    plot(t/60, (evader_radius - const.R_EARTH) / 1e3, 'Color', [0 0.8 0], 'LineWidth', 2);
    plot(t/60, (pursuer_radius - const.R_EARTH) / 1e3, 'Color', [0.8 0 0], 'LineWidth', 2);
    xlabel('Time [min]'); ylabel('Altitude [km]'); title('Altitude Profiles');

    subplot(2, 3, 3);
    hold on; grid on; box on;
    inter_dist = vecnorm(evader - pursuer, 2, 2);
    plot(t/60, inter_dist/1e3, 'b-', 'LineWidth', 2);
    yline(const.CAPTURE_RADIUS/1e3, 'r--');
    yline(const.EVASION_RADIUS/1e3, 'g--');
    xlabel('Time [min]'); ylabel('Distance [km]'); title('Inter-Spacecraft Distance');

    subplot(2, 3, 5);
    hold on; grid on; box on;
    evader_period = 2*pi*sqrt(mean(evader_radius).^3/const.MU_EARTH) / 60;
    pursuer_period = 2*pi*sqrt(mean(pursuer_radius).^3/const.MU_EARTH) / 60;
    evader_alt_mean = mean((evader_radius - const.R_EARTH) / 1e3);
    pursuer_alt_mean = mean((pursuer_radius - const.R_EARTH) / 1e3);
    evader_ecc = std(evader_radius) / mean(evader_radius);
    pursuer_ecc = std(pursuer_radius) / mean(pursuer_radius);
    bar([evader_period, pursuer_period; evader_alt_mean, pursuer_alt_mean; evader_ecc, pursuer_ecc]');
    set(gca, 'XTickLabel', {'Period [min]', 'Mean Alt [km]', 'Eccentricity'});
    legend('Evader', 'Pursuer', 'Location', 'best');
    title('Orbital Parameters');

    subplot(2, 3, 6);
    hold on; grid on; box on;
    evader_lat = asind(evader(:,3) ./ evader_radius);
    evader_lon = atan2d(evader(:,2), evader(:,1));
    pursuer_lat = asind(pursuer(:,3) ./ pursuer_radius);
    pursuer_lon = atan2d(pursuer(:,2), pursuer(:,1));
    plot(evader_lon, evader_lat, 'Color', [0 0.8 0], 'LineWidth', 1.5);
    plot(pursuer_lon, pursuer_lat, 'Color', [0.8 0 0], 'LineWidth', 1.5);
    xlabel('Longitude [deg]'); ylabel('Latitude [deg]'); title('Ground Track');
    xlim([-180, 180]); ylim([-90, 90]);
end

function plot_delta_v_diagnostics(step_dv, evader_cumulative, pursuer_cumulative, dt_seconds, time_axis)
    if isempty(time_axis)
        time_axis = (0:numel(step_dv.evader_magnitude)-1)';
    end

    subplot(3, 3, 1);
    hold on; grid on; box on;
    plot(time_axis, step_dv.evader_magnitude, 'Color', [0 0.8 0], 'LineWidth', 1.5);
    plot(time_axis, step_dv.pursuer_magnitude, 'Color', [0.8 0 0], 'LineWidth', 1.5);
    xlabel('Time [s]'); ylabel('|ΔV| [m/s]'); title('Instantaneous ΔV Magnitude');

    subplot(3, 3, 2);
    hold on; grid on; box on;
    plot(time_axis, evader_cumulative, 'Color', [0 0.8 0], 'LineWidth', 2);
    plot(time_axis, pursuer_cumulative, 'Color', [0.8 0 0], 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Cumulative ΔV [m/s]'); title('Total ΔV Consumption');

    subplot(3, 3, 3);
    hold on; grid on; box on;
    ratio = pursuer_cumulative ./ max(evader_cumulative, eps);
    plot(time_axis, ratio, 'k-', 'LineWidth', 2);
    yline(1, 'b--');
    xlabel('Time [s]'); ylabel('Pursuer / Evader'); title('ΔV Efficiency Ratio');

    subplot(3, 3, 4);
    hold on; grid on; box on;
    plot(time_axis, step_dv.evader_vectors(:,1), 'r-');
    plot(time_axis, step_dv.evader_vectors(:,2), 'g-');
    plot(time_axis, step_dv.evader_vectors(:,3), 'b-');
    xlabel('Time [s]'); ylabel('ΔV Component [m/s]'); title('Evader ΔV Components');

    subplot(3, 3, 5);
    hold on; grid on; box on;
    plot(time_axis, step_dv.pursuer_vectors(:,1), 'r-');
    plot(time_axis, step_dv.pursuer_vectors(:,2), 'g-');
    plot(time_axis, step_dv.pursuer_vectors(:,3), 'b-');
    xlabel('Time [s]'); ylabel('ΔV Component [m/s]'); title('Pursuer ΔV Components');

    subplot(3, 3, 6);
    hold on; grid on; box on;
    sample_idx = 1:max(1, floor(numel(time_axis)/25)):numel(time_axis);
    sample_idx = sample_idx(:);
    origin = zeros(numel(sample_idx), 1);
    quiver3(origin, origin, origin, ...
            step_dv.evader_vectors(sample_idx,1), step_dv.evader_vectors(sample_idx,2), step_dv.evader_vectors(sample_idx,3), ...
            'Color', [0 0.8 0], 'LineWidth', 1.2);
    quiver3(origin, origin, origin, ...
            step_dv.pursuer_vectors(sample_idx,1), step_dv.pursuer_vectors(sample_idx,2), step_dv.pursuer_vectors(sample_idx,3), ...
            'Color', [0.8 0 0], 'LineWidth', 1.2);
    xlabel('ΔV_x'); ylabel('ΔV_y'); zlabel('ΔV_z'); title('ΔV Directions'); view(45, 25); axis equal;

    subplot(3, 3, 7);
    hold on;
    histogram(step_dv.evader_magnitude, 30, 'FaceColor', [0 0.8 0], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    histogram(step_dv.pursuer_magnitude, 30, 'FaceColor', [0.8 0 0], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    xlabel('|ΔV| [m/s]'); ylabel('Frequency'); title('ΔV Distribution');

    subplot(3, 3, 8);
    hold on; grid on; box on;
    dv_matrix = [step_dv.evader_magnitude, step_dv.pursuer_magnitude];
    dv_labels = {'Evader', 'Pursuer'};
    try
        boxplot(dv_matrix, dv_labels);
        ylabel('|ΔV| [m/s]');
        title('ΔV Statistical Comparison');
        grid on;
    catch boxplotErr %#ok<NASGU>
        cla;
        hold on; grid on; box on;
        plot_delta_v_boxplot_fallback({step_dv.evader_magnitude, step_dv.pursuer_magnitude}, dv_labels);
        ylabel('|ΔV| [m/s]');
        title('ΔV Statistical Comparison (Fallback)');
    end

    subplot(3, 3, 9);
    hold on; grid on; box on;
    window = max(1, floor(numel(step_dv.evader_magnitude)/20));
    plot(time_axis, movmean(step_dv.evader_magnitude, window), 'Color', [0 0.8 0], 'LineWidth', 2);
    plot(time_axis, movmean(step_dv.pursuer_magnitude, window), 'Color', [0.8 0 0], 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Smoothed |ΔV| [m/s]'); title(sprintf('Moving Average (window = %d)', window));

    if ~isempty(dt_seconds)
        sgtitle(sprintf('ΔV Diagnostics (Δt ≈ %.1f s)', dt_seconds));
    end
end

function plot_delta_v_boxplot_fallback(dv_sets, dv_labels)
    dv_colors = [0 0.8 0; 0.8 0 0];
    box_width = 0.35;
    for idx = 1:numel(dv_sets)
        data_vec = dv_sets{idx};
        data_vec = data_vec(~isnan(data_vec));
        if isempty(data_vec)
            continue;
        end
        data_vec = sort(data_vec(:));
        n = numel(data_vec);
        if n == 1
            quartiles = repmat(data_vec, 1, 3);
        else
            positions = (n - 1) * [0.25, 0.5, 0.75] + 1;
            quartiles = interp1(1:n, data_vec, positions, 'linear');
        end
        q1 = quartiles(1);
        q2 = quartiles(2);
        q3 = quartiles(3);
        iqr_val = q3 - q1;
        whisker_low = max(data_vec(data_vec >= q1 - 1.5 * iqr_val), [], 'omitnan');
        whisker_high = min(data_vec(data_vec <= q3 + 1.5 * iqr_val), [], 'omitnan');
        if isempty(whisker_low)
            whisker_low = data_vec(1);
        end
        if isempty(whisker_high)
            whisker_high = data_vec(end);
        end
        x_pos = idx;
        base_color = dv_colors(min(idx, size(dv_colors, 1)), :);
        fill([x_pos - box_width, x_pos - box_width, x_pos + box_width, x_pos + box_width], ...
             [q1, q3, q3, q1], base_color, 'FaceAlpha', 0.25, 'EdgeColor', base_color);
        plot([x_pos - box_width, x_pos + box_width], [q2, q2], 'Color', base_color, 'LineWidth', 2);
        plot([x_pos, x_pos], [whisker_low, q1], 'Color', base_color, 'LineWidth', 1.2);
        plot([x_pos, x_pos], [q3, whisker_high], 'Color', base_color, 'LineWidth', 1.2);
        plot([x_pos - box_width/2, x_pos + box_width/2], [whisker_low, whisker_low], 'Color', base_color, 'LineWidth', 1.2);
        plot([x_pos - box_width/2, x_pos + box_width/2], [whisker_high, whisker_high], 'Color', base_color, 'LineWidth', 1.2);
        scatter(repmat(x_pos, size(data_vec)), data_vec, 8, 'MarkerEdgeColor', base_color, 'Marker', '.');
    end
    xlim([0.5, numel(dv_sets) + 0.5]);
    set(gca, 'XTick', 1:numel(dv_sets), 'XTickLabel', dv_labels);
end

function plot_mission_statistics(rel_distance, rel_velocity, evader_cumulative, pursuer_cumulative, per_episode_dv, progress, stats, const)
    subplot(2, 3, 1);
    hold on; grid on; box on;
    data = [];
    labels = {};
    if ~isempty(rel_distance)
        data = [data; rel_distance(end)/1e3, const.CAPTURE_RADIUS/1e3, const.EVASION_RADIUS/1e3];
        labels{end+1} = 'Distance [km]'; %#ok<AGROW>
    end
    if ~isempty(evader_cumulative)
        data = [data; evader_cumulative(end), pursuer_cumulative(end), 0]; %#ok<AGROW>
        labels{end+1} = 'Total ΔV [m/s]'; %#ok<AGROW>
    end
    if ~isempty(rel_velocity)
        data = [data; mean(rel_velocity), max(rel_velocity), min(rel_velocity)]; %#ok<AGROW>
        labels{end+1} = 'Velocity [m/s]'; %#ok<AGROW>
    end
    if isempty(data)
        axis off; text(0.5, 0.5, 'Insufficient data for summary metrics', 'HorizontalAlignment', 'center');
    else
        bar(data);
        set(gca, 'XTickLabel', labels);
        legend('Final / Evader', 'Capture / Pursuer', 'Evasion / Stats', 'Location', 'best');
        title('Key Performance Metrics');
    end

    subplot(2, 3, 2);
    if ~isempty(rel_distance) && ~isempty(evader_cumulative)
        segments = 5;
        idx = round(linspace(1, numel(rel_distance), segments+1));
        segment_distance = zeros(segments,1);
        for k = 1:segments
            segment_distance(k) = mean(rel_distance(idx(k):idx(k+1)-1));
        end
        plot(1:segments, segment_distance/1e3, 'b-o', 'LineWidth', 2);
        ylabel('Mean distance [km]'); xlabel('Segment'); title('Distance by Segment'); grid on;
    else
        axis off;
    end

    subplot(2, 3, 3);
    if ~isempty(per_episode_dv)
        episode_idx = 1:numel(per_episode_dv);
        bar(episode_idx, per_episode_dv, 'FaceColor', [0 0.6 0.2]);
        xlabel('Episode'); ylabel('Total ΔV [m/s]');
        title('Per-Episode Evader ΔV'); grid on;
    else
        axis off; text(0.5, 0.5, 'Per-episode ΔV not available', 'HorizontalAlignment', 'center');
    end

    subplot(2, 3, 4);
    if isstruct(progress) && isfield(progress, 'success_rates') && ~isempty(progress.success_rates)
        plot(progress.success_rates, 'LineWidth', 2);
        xlabel('Evaluation Index'); ylabel('Success Rate'); title('Success Rate Trend'); grid on;
    else
        axis off; text(0.5, 0.5, 'Success-rate data unavailable', 'HorizontalAlignment', 'center');
    end

    subplot(2, 3, 5);
    if isstruct(stats) && isfield(stats, 'episodes_info') && ~isempty(stats.episodes_info)
        outcomes = string({stats.episodes_info.outcome});
        categories = categorical(outcomes);
        histogram(categories);
        title('Recent Episode Outcomes');
    else
        axis off;
    end

    subplot(2, 3, 6);
    axis off;
    lines = {};
    if isstruct(stats)
        if isfield(stats, 'total_episodes')
            lines{end+1} = sprintf('Total episodes: %d', stats.total_episodes);
        end
        if isfield(stats, 'final_success_rate')
            lines{end+1} = sprintf('Final success rate: %.2f %%', 100*stats.final_success_rate);
        end
        if isfield(stats, 'captures')
            lines{end+1} = sprintf('Captures: %d', stats.captures);
        end
        if isfield(stats, 'fuel_depleted')
            lines{end+1} = sprintf('Fuel depleted: %d', stats.fuel_depleted);
        end
    end
    if isempty(lines)
        text(0.5, 0.5, 'No summary statistics available', 'HorizontalAlignment', 'center');
    else
        text(0.05, 0.9, strjoin(lines, '\n'), 'Units', 'normalized', 'FontName', 'monospace');
    end
end

function save_figure_if_needed(fig, name, export_dir, config)
    if ~config.save_figures
        return;
    end
    for fmt = config.figure_formats
        filepath = fullfile(export_dir, sprintf('%s.%s', name, fmt{1}));
        exportgraphics(fig, filepath, 'Resolution', 200);
    end
end
