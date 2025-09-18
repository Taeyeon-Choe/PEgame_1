%% ========================================================================
%  MATLAB Analysis Automation for Pursuit-Evasion Training Logs
%  Generates mission diagnostics directly from Python training outputs.
% ========================================================================

clearvars; close all; clc;
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
config.logs_root = './logs';       % Root directory for training logs
config.target_run = '';            % Specify run folder or leave empty for latest
config.experiment_filter = '*';    % Wildcard filter (e.g., 'interactive_training*')
config.save_figures = false;       % Save figures automatically
config.figure_formats = {'png'};   % Formats used when saving figures
config.output_dir = '';            % Optional override for figure output path
config.verbose = true;             % Display progress information

%% ------------------------------------------------------------------------
%  SELECT LOG DIRECTORY
%% ------------------------------------------------------------------------
[run_info, config] = resolve_run_directory(config);
plots_dir = fullfile(run_info.path, 'plots');
analysis_dir = fullfile(run_info.path, 'analysis');
if isempty(config.output_dir)
    export_dir = fullfile(run_info.path, 'matlab_analysis');
else
    export_dir = config.output_dir;
end
if config.save_figures && ~exist(export_dir, 'dir')
    mkdir(export_dir);
end

if config.verbose
    fprintf('=====================================\n');
    fprintf('Selected training run : %s\n', run_info.name);
    fprintf('Log directory         : %s\n', run_info.path);
    fprintf('Output directory      : %s\n', export_dir);
    fprintf('=====================================\n\n');
end

%% ------------------------------------------------------------------------
%  LOAD AVAILABLE DATA SETS
%% ------------------------------------------------------------------------
const.R_EARTH = 6378e3;                % Earth radius [m]
const.MU_EARTH = 3.986004418e14;       % Earth gravitational parameter [m^3/s^2]
const.CAPTURE_RADIUS = 1e3;            % Capture distance [m]
const.EVASION_RADIUS = 5e3;            % Evasion distance [m]

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

%% ------------------------------------------------------------------------
%  BUILD PRIMARY EPISODE DATA (FIRST COMPLETED EPISODE)
%% ------------------------------------------------------------------------
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
else
    trajectory = struct();
end

% Fallback to derived distances if not provided
if ~isempty(trajectory) && (~isfield(trajectory, 'distances') || isempty(trajectory.distances))
    trajectory.distances = sqrt(trajectory.x.^2 + trajectory.y.^2 + trajectory.z.^2);
end

if ~isempty(trajectory) && (~isfield(trajectory, 'times') || isempty(trajectory.times))
    n_points = numel(trajectory.distances);
    trajectory.times = (0:n_points-1)';
end

% Delta-V per step
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

% Episode summary list for per-episode analytics
per_episode_dv = [];
if isstruct(stats) && isfield(stats, 'evader_delta_v') && ~isempty(stats.evader_delta_v)
    per_episode_dv = stats.evader_delta_v(:);
end
if isempty(per_episode_dv) && isstruct(delta_v_table)
    if isfield(delta_v_table, 'delta_v')
        per_episode_dv = delta_v_table.delta_v(:);
    elseif isfield(delta_v_table, 'Var2')
        per_episode_dv = delta_v_table.Var2(:);
    end
end

mission_outcome = 'UNKNOWN';
if ~isempty(primary_episode) && isfield(primary_episode, 'outcome')
    mission_outcome = upper(string(primary_episode.outcome));
elseif isstruct(stats) && isfield(stats, 'episodes_info') && ~isempty(stats.episodes_info)
    mission_outcome = upper(string(stats.episodes_info(end).outcome));
end

%% ------------------------------------------------------------------------
%  DERIVED METRICS
%% ------------------------------------------------------------------------
has_trajectory = isfield(trajectory, 'x') && ~isempty(trajectory.x);
has_deltav = isfield(step_dv, 'evader_magnitude') && ~isempty(step_dv.evader_magnitude);
has_ephemeris = ~isempty(ephemeris) && all(isfield(ephemeris, {'t', 'evader', 'pursuer'}));

if has_trajectory
    rel_distance = trajectory.distances(:);
    rel_velocity = sqrt(trajectory.vx.^2 + trajectory.vy.^2 + trajectory.vz.^2);
    time_axis = trajectory.times;
else
    rel_distance = [];
    rel_velocity = [];
    time_axis = [];
end

dt_seconds = []; 
if numel(time_axis) > 1
    dt_seconds = median(diff(time_axis));
end

if has_deltav
    evader_cumulative = cumsum(step_dv.evader_magnitude);
    pursuer_cumulative = cumsum(step_dv.pursuer_magnitude);
else
    evader_cumulative = [];
    pursuer_cumulative = [];
end

%% ------------------------------------------------------------------------
%  FIGURE 1: RELATIVE TRAJECTORY (LVLH)
%% ------------------------------------------------------------------------
fig1 = figure('Name', 'Relative Trajectory (LVLH)', 'Position', [80, 80, 1400, 900], 'Color', 'white');

if has_trajectory
    subplot(2, 3, [1, 4]);
    hold on; grid on; box on;
    c = linspace(0, 1, numel(trajectory.x));
    surface([trajectory.x; nan], [trajectory.y; nan], [trajectory.z; nan], [c'; nan], ...
            'EdgeColor', 'interp', 'FaceColor', 'none', 'LineWidth', 2.5);
    plot3(0, 0, 0, 'o', 'MarkerSize', 10, 'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'k');
    plot3(trajectory.x(1), trajectory.y(1), trajectory.z(1), '^', 'MarkerSize', 9, 'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'k');
    plot3(trajectory.x(end), trajectory.y(end), trajectory.z(end), 's', 'MarkerSize', 9, 'MarkerFaceColor', [0.2 0.2 0.2], 'MarkerEdgeColor', 'k');
    draw_zone_spheres(const);
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    title(sprintf('Relative Trajectory - Outcome: %s', mission_outcome));
    axis equal; view(45, 25);
    colorbar('Label', 'Step progression');
else
    subplot(2, 3, [1, 4]);
    axis off;
    text(0.5, 0.5, 'Relative position data not available', 'HorizontalAlignment', 'center');
end

t = time_axis;
if isempty(t)
    t = (0:numel(rel_distance)-1)';
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
    plot(t, trajectory.vx, 'r-');
    plot(t, trajectory.vy, 'g-');
    plot(t, trajectory.vz, 'b-');
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
    scatter(trajectory.x, trajectory.vx, 20, c, 'filled');
    xlabel('Position X [m]'); ylabel('Velocity V_x [m/s]');
    title('Phase Space (X vs V_x)');
    colorbar('Label', 'Step progression');
else
    axis off; text(0.5, 0.5, 'Phase space unavailable', 'HorizontalAlignment', 'center');
end

subplot(2, 3, 6);
if ~isempty(rel_distance)
    hold on; grid on; box on;
    distance_rate = [0; diff(rel_distance)];
    plot(t, distance_rate, 'b-', 'LineWidth', 1.5);
    yline(0, 'k--');
    xlabel('Time [s]'); ylabel('ΔDistance [m]');
    title('Closing / Opening Rate');
else
    axis off;
end

save_figure_if_needed(fig1, 'relative_trajectory', export_dir, config);

%% ------------------------------------------------------------------------
%  FIGURE 2: ECI TRAJECTORY (FINAL EPISODE)
%% ------------------------------------------------------------------------
fig2 = figure('Name', 'ECI Trajectory', 'Position', [100, 100, 1400, 900], 'Color', 'white');
if has_ephemeris
    plot_eci_diagnostics(ephemeris, const);
else
    axis off;
    text(0.5, 0.5, 'ECI data not available (enable EphemerisLoggerCallback)', 'HorizontalAlignment', 'center');
end
save_figure_if_needed(fig2, 'eci_trajectory', export_dir, config);

%% ------------------------------------------------------------------------
%  FIGURE 3: STEPWISE DELTA-V ANALYSIS
%% ------------------------------------------------------------------------
fig3 = figure('Name', 'Delta-V Analysis', 'Position', [120, 120, 1400, 900], 'Color', 'white');
if has_deltav
    plot_delta_v_diagnostics(step_dv, evader_cumulative, pursuer_cumulative, dt_seconds, t);
else
    axis off;
    text(0.5, 0.5, 'Stepwise delta-v data not available', 'HorizontalAlignment', 'center');
end
save_figure_if_needed(fig3, 'delta_v_analysis', export_dir, config);

%% ------------------------------------------------------------------------
%  FIGURE 4: TRAINING-WIDE METRICS
%% ------------------------------------------------------------------------
fig4 = figure('Name', 'Mission Statistics', 'Position', [140, 140, 1200, 800], 'Color', 'white');
plot_mission_statistics(rel_distance, rel_velocity, evader_cumulative, pursuer_cumulative, per_episode_dv, progress, stats, const);
save_figure_if_needed(fig4, 'mission_statistics', export_dir, config);

if config.verbose
    fprintf('\nAnalysis complete.\n');
    if config.save_figures
        fprintf('Figures saved to: %s\n', export_dir);
    end
end

%% ========================================================================
%  HELPER FUNCTIONS
% ========================================================================
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
    quiver3(zeros(size(sample_idx)), zeros(size(sample_idx)), zeros(size(sample_idx)), ...
            step_dv.evader_vectors(sample_idx,1), step_dv.evader_vectors(sample_idx,2), step_dv.evader_vectors(sample_idx,3), ...
            'Color', [0 0.8 0], 'LineWidth', 1.2);
    quiver3(zeros(size(sample_idx)), zeros(size(sample_idx)), zeros(size(sample_idx)), ...
            step_dv.pursuer_vectors(sample_idx,1), step_dv.pursuer_vectors(sample_idx,2), step_dv.pursuer_vectors(sample_idx,3), ...
            'Color', [0.8 0 0], 'LineWidth', 1.2);
    xlabel('ΔV_x'); ylabel('ΔV_y'); zlabel('ΔV_z'); title('ΔV Directions'); view(45, 25); axis equal;

    subplot(3, 3, 7);
    hold on;
    histogram(step_dv.evader_magnitude, 30, 'FaceColor', [0 0.8 0], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    histogram(step_dv.pursuer_magnitude, 30, 'FaceColor', [0.8 0 0], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
    xlabel('|ΔV| [m/s]'); ylabel('Frequency'); title('ΔV Distribution');

    subplot(3, 3, 8);
    boxplot([step_dv.evader_magnitude, step_dv.pursuer_magnitude], {'Evader', 'Pursuer'});
    ylabel('|ΔV| [m/s]'); title('ΔV Statistical Comparison'); grid on;

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
