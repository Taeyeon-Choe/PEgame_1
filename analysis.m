%% ========================================================================
%  Complete Spacecraft Pursuit-Evasion Trajectory Analysis Suite
%  Version: 2.0
%  Description: Comprehensive MATLAB visualization and analysis tool for
%               spacecraft pursuit-evasion scenarios from MAT data files
%  Author: Spacecraft Dynamics Analysis Team
%  Date: 2025
% =========================================================================

%% Initialize Environment
clear all; close all; clc;
format long g;

% Add custom colormaps
addpath(genpath(pwd));

% Set default plot properties
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontSize', 12);
set(0, 'DefaultAxesGridAlpha', 0.3);
set(0, 'DefaultAxesXGrid', 'on');
set(0, 'DefaultAxesYGrid', 'on');
set(0, 'DefaultAxesZGrid', 'on');

%% ========================================================================
%  CONFIGURATION SECTION
% =========================================================================

% File paths configuration
config.base_path = './results/';  % Adjust to your data directory
config.scenario_id = 'test_1';    % Scenario identifier
config.save_figures = false;      % Save figures to disk
config.save_format = {'png', 'fig'};  % Save formats
config.animation.enable = false;   % Enable animations
config.animation.fps = 30;        % Animation frame rate
config.animation.save = false;    % Save animation as video

% Physical constants
const.R_EARTH = 6378000;          % Earth radius [m]
const.MU_EARTH = 3.986004418e14;  % Earth gravitational parameter [m^3/s^2]
const.CAPTURE_RADIUS = 1000;      % Capture distance [m]
const.EVASION_RADIUS = 5000;      % Evasion distance [m]

% Plotting parameters
plot_params.colormap = 'jet';
plot_params.evader_color = [0 0.8 0];      % Green
plot_params.pursuer_color = [0.8 0 0];     % Red
plot_params.earth_color = [0.4 0.6 1];     % Light blue
plot_params.trajectory_alpha = 0.8;
plot_params.sphere_resolution = 30;
plot_params.marker_size = 10;

%% ========================================================================
%  DATA LOADING SECTION
% =========================================================================

fprintf('=====================================\n');
fprintf('   Spacecraft Trajectory Analysis   \n');
fprintf('=====================================\n\n');
fprintf('Loading data files...\n');

% Construct file paths
files.trajectory = fullfile(config.base_path, sprintf('%s_trajectory_data.mat', config.scenario_id));
files.eci = fullfile(config.base_path, sprintf('%s_eci.mat', config.scenario_id));
files.deltav = fullfile(config.base_path, sprintf('%s_deltav.mat', config.scenario_id));

% Load trajectory data (relative position)
if exist(files.trajectory, 'file')
    data.trajectory = load(files.trajectory);
    fprintf('✓ Loaded trajectory data\n');
    has_trajectory = true;
else
    warning('Trajectory file not found: %s', files.trajectory);
    has_trajectory = false;
    % Generate sample data for demonstration
    t = linspace(0, 1000, 100)';
    data.trajectory.x = 5000 * exp(-t/500) .* cos(2*pi*t/100);
    data.trajectory.y = 5000 * exp(-t/500) .* sin(2*pi*t/100);
    data.trajectory.z = 1000 * sin(2*pi*t/200);
    data.trajectory.vx = gradient(data.trajectory.x);
    data.trajectory.vy = gradient(data.trajectory.y);
    data.trajectory.vz = gradient(data.trajectory.z);
end

% Load ECI data
if exist(files.eci, 'file')
    data.eci = load(files.eci);
    fprintf('✓ Loaded ECI data\n');
    has_eci = true;
else
    warning('ECI file not found: %s', files.eci);
    has_eci = false;
    % Generate synthetic ECI data if needed
    n_points = length(data.trajectory.x);
    data.eci.t = linspace(0, 1000, n_points)';
    altitude = 400000; % 400 km altitude
    r = const.R_EARTH + altitude;
    omega = sqrt(const.MU_EARTH / r^3);
    data.eci.evader_x = r * cos(omega * data.eci.t);
    data.eci.evader_y = r * sin(omega * data.eci.t);
    data.eci.evader_z = zeros(n_points, 1);
    data.eci.pursuer_x = data.eci.evader_x + data.trajectory.x;
    data.eci.pursuer_y = data.eci.evader_y + data.trajectory.y;
    data.eci.pursuer_z = data.eci.evader_z + data.trajectory.z;
end

% Load delta-v data
if exist(files.deltav, 'file')
    data.deltav = load(files.deltav);
    fprintf('✓ Loaded delta-v data\n');
    has_deltav = true;
else
    warning('Delta-v file not found: %s', files.deltav);
    has_deltav = false;
    % Generate synthetic delta-v data
    n_points = length(data.trajectory.x);
    data.deltav.time = linspace(0, 1000, n_points)';
    data.deltav.evader_dv = 0.01 * randn(n_points, 3) .* exp(-data.deltav.time/500);
    data.deltav.pursuer_dv = 0.015 * randn(n_points, 3) .* exp(-data.deltav.time/500);
    data.deltav.evader_dv_norm = sqrt(sum(data.deltav.evader_dv.^2, 2));
    data.deltav.pursuer_dv_norm = sqrt(sum(data.deltav.pursuer_dv.^2, 2));
    data.deltav.evader_cumulative = cumsum(data.deltav.evader_dv_norm);
    data.deltav.pursuer_cumulative = cumsum(data.deltav.pursuer_dv_norm);
end

fprintf('\nData loading complete.\n\n');

%% ========================================================================
%  DATA PREPROCESSING
% =========================================================================

% Calculate derived quantities
n_points = length(data.trajectory.x);
time_steps = 1:n_points;

% Relative distance
rel_distance = sqrt(data.trajectory.x.^2 + data.trajectory.y.^2 + data.trajectory.z.^2);

% Relative velocity magnitude
if isfield(data.trajectory, 'vx')
    rel_velocity = sqrt(data.trajectory.vx.^2 + data.trajectory.vy.^2 + data.trajectory.vz.^2);
else
    rel_velocity = [0; sqrt(diff(data.trajectory.x).^2 + diff(data.trajectory.y).^2 + diff(data.trajectory.z).^2)];
end

% Mission outcome determination
if rel_distance(end) < const.CAPTURE_RADIUS
    mission_outcome = 'CAPTURE';
    outcome_color = plot_params.pursuer_color;
elseif rel_distance(end) > const.EVASION_RADIUS
    mission_outcome = 'EVASION';
    outcome_color = plot_params.evader_color;
else
    mission_outcome = 'ONGOING';
    outcome_color = [0.5 0.5 0.5];
end

%% ========================================================================
%  FIGURE 1: COMPREHENSIVE RELATIVE TRAJECTORY ANALYSIS
% =========================================================================

fig1 = figure('Name', 'Relative Trajectory Analysis', ...
              'Position', [50, 50, 1400, 900], ...
              'Color', 'white');

% --- Subplot 1: 3D Trajectory ---
ax1 = subplot(2, 3, [1, 4]);
hold on; grid on; box on;

% Plot trajectory with gradient color
colormap(ax1, plot_params.colormap);
c = linspace(0, 1, n_points);
patch([data.trajectory.x; nan], [data.trajectory.y; nan], [data.trajectory.z; nan], ...
      [c'; nan], 'EdgeColor', 'interp', 'LineWidth', 2.5, ...
      'FaceColor', 'none', 'EdgeAlpha', plot_params.trajectory_alpha);

% Mark key points
plot3(0, 0, 0, 'o', 'MarkerSize', plot_params.marker_size, ...
      'MarkerFaceColor', plot_params.evader_color, ...
      'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
plot3(data.trajectory.x(1), data.trajectory.y(1), data.trajectory.z(1), ...
      '^', 'MarkerSize', plot_params.marker_size, ...
      'MarkerFaceColor', plot_params.pursuer_color, ...
      'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
plot3(data.trajectory.x(end), data.trajectory.y(end), data.trajectory.z(end), ...
      's', 'MarkerSize', plot_params.marker_size, ...
      'MarkerFaceColor', outcome_color, ...
      'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

% Add capture and evasion zones
[xs, ys, zs] = sphere(plot_params.sphere_resolution);

% Capture sphere
surf(xs*const.CAPTURE_RADIUS, ys*const.CAPTURE_RADIUS, zs*const.CAPTURE_RADIUS, ...
     'FaceColor', plot_params.pursuer_color, 'EdgeColor', 'none', ...
     'FaceAlpha', 0.15, 'DisplayName', 'Capture Zone');

% Evasion sphere
surf(xs*const.EVASION_RADIUS, ys*const.EVASION_RADIUS, zs*const.EVASION_RADIUS, ...
     'FaceColor', plot_params.evader_color, 'EdgeColor', 'none', ...
     'FaceAlpha', 0.1, 'DisplayName', 'Evasion Zone');

% Add velocity vectors at key points
if isfield(data.trajectory, 'vx')
    indices = [1, round(n_points/4), round(n_points/2), round(3*n_points/4), n_points];
    for idx = indices
        if idx <= n_points
            quiver3(data.trajectory.x(idx), data.trajectory.y(idx), data.trajectory.z(idx), ...
                   data.trajectory.vx(idx), data.trajectory.vy(idx), data.trajectory.vz(idx), ...
                   500, 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
        end
    end
end

xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title(sprintf('3D Relative Trajectory - Outcome: %s', mission_outcome));
legend('Trajectory', 'Evader', 'Pursuer Start', 'Final Position', ...
       'Location', 'best', 'FontSize', 10);
view(45, 25);
axis equal;
colorbar('Label', 'Time Progress');

% --- Subplot 2: Distance Evolution ---
ax2 = subplot(2, 3, 2);
hold on; grid on; box on;

plot(time_steps, rel_distance, 'b-', 'LineWidth', 2);
plot(time_steps, const.CAPTURE_RADIUS * ones(size(time_steps)), ...
     'r--', 'LineWidth', 1.5);
plot(time_steps, const.EVASION_RADIUS * ones(size(time_steps)), ...
     'g--', 'LineWidth', 1.5);

% Shade critical regions
fill([time_steps, fliplr(time_steps)], ...
     [zeros(size(time_steps)), const.CAPTURE_RADIUS * ones(size(time_steps))], ...
     plot_params.pursuer_color, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
fill([time_steps, fliplr(time_steps)], ...
     [const.EVASION_RADIUS * ones(size(time_steps)), 10000 * ones(size(time_steps))], ...
     plot_params.evader_color, 'FaceAlpha', 0.1, 'EdgeColor', 'none');

xlabel('Time Step'); ylabel('Distance [m]');
title('Relative Distance Evolution');
legend('Distance', 'Capture Threshold', 'Evasion Threshold', ...
       'Location', 'best');
xlim([1, n_points]);
ylim([0, max(rel_distance)*1.1]);

% --- Subplot 3: Velocity Components ---
ax3 = subplot(2, 3, 3);
if isfield(data.trajectory, 'vx')
    hold on; grid on; box on;
    plot(time_steps, data.trajectory.vx, 'r-', 'LineWidth', 1.5);
    plot(time_steps, data.trajectory.vy, 'g-', 'LineWidth', 1.5);
    plot(time_steps, data.trajectory.vz, 'b-', 'LineWidth', 1.5);
    plot(time_steps, rel_velocity, 'k--', 'LineWidth', 2);
    xlabel('Time Step'); ylabel('Velocity [m/s]');
    title('Relative Velocity Components');
    legend('V_x', 'V_y', 'V_z', '|V|', 'Location', 'best');
else
    text(0.5, 0.5, 'Velocity data not available', ...
         'HorizontalAlignment', 'center', 'Units', 'normalized');
    axis off;
end

% --- Subplot 5: Phase Space (X-Vx) ---
ax5 = subplot(2, 3, 5);
if isfield(data.trajectory, 'vx')
    hold on; grid on; box on;
    scatter(data.trajectory.x, data.trajectory.vx, 20, c, 'filled');
    plot(data.trajectory.x(1), data.trajectory.vx(1), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    plot(data.trajectory.x(end), data.trajectory.vx(end), 'ks', 'MarkerSize', 8, 'LineWidth', 2);
    xlabel('Position X [m]'); ylabel('Velocity V_x [m/s]');
    title('Phase Space (X-V_x)');
    colormap(ax5, plot_params.colormap);
    colorbar('Label', 'Time Progress');
else
    text(0.5, 0.5, 'Phase space requires velocity data', ...
         'HorizontalAlignment', 'center', 'Units', 'normalized');
    axis off;
end

% --- Subplot 6: Distance Rate ---
ax6 = subplot(2, 3, 6);
hold on; grid on; box on;
distance_rate = [0; diff(rel_distance)];
plot(time_steps, distance_rate, 'b-', 'LineWidth', 1.5);
plot(time_steps, zeros(size(time_steps)), 'k--', 'LineWidth', 1);
fill([time_steps, fliplr(time_steps)], ...
     [distance_rate', zeros(size(time_steps))], ...
     'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
xlabel('Time Step'); ylabel('Distance Rate [m/step]');
title('Closing/Opening Rate');
legend('Rate', 'Zero Line', 'Location', 'best');

%% ========================================================================
%  FIGURE 2: ECI TRAJECTORY AND ORBITAL ANALYSIS
% =========================================================================

if has_eci
    fig2 = figure('Name', 'ECI Orbital Analysis', ...
                  'Position', [100, 75, 1400, 900], ...
                  'Color', 'white');
    
    % --- Subplot 1: 3D ECI Trajectories ---
    ax1 = subplot(2, 3, [1, 4]);
    hold on; grid on; box on;
    
    % Earth representation
    [xe, ye, ze] = sphere(plot_params.sphere_resolution);
    earth = surf(xe*const.R_EARTH, ye*const.R_EARTH, ze*const.R_EARTH);
    set(earth, 'FaceColor', plot_params.earth_color, ...
               'EdgeColor', 'none', 'FaceAlpha', 0.7, ...
               'FaceLighting', 'gouraud', 'AmbientStrength', 0.5);
    light('Position', [1 0 1], 'Style', 'infinite');
    
    % Plot trajectories
    plot3(data.eci.evader_x, data.eci.evader_y, data.eci.evader_z, ...
          'Color', plot_params.evader_color, 'LineWidth', 2.5);
    plot3(data.eci.pursuer_x, data.eci.pursuer_y, data.eci.pursuer_z, ...
          'Color', plot_params.pursuer_color, 'LineWidth', 2.5);
    
    % Mark orbital positions
    plot3(data.eci.evader_x(1), data.eci.evader_y(1), data.eci.evader_z(1), ...
          'o', 'MarkerSize', 12, 'MarkerFaceColor', plot_params.evader_color, ...
          'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    plot3(data.eci.pursuer_x(1), data.eci.pursuer_y(1), data.eci.pursuer_z(1), ...
          '^', 'MarkerSize', 12, 'MarkerFaceColor', plot_params.pursuer_color, ...
          'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    title('Earth-Centered Inertial (ECI) Trajectories');
    legend('Earth', 'Evader Orbit', 'Pursuer Orbit', ...
           'Evader Start', 'Pursuer Start', 'Location', 'best');
    view(45, 25);
    axis equal;
    
    % --- Subplot 2: Altitude Profiles ---
    ax2 = subplot(2, 3, 2);
    hold on; grid on; box on;
    
    evader_radius = sqrt(data.eci.evader_x.^2 + data.eci.evader_y.^2 + data.eci.evader_z.^2);
    pursuer_radius = sqrt(data.eci.pursuer_x.^2 + data.eci.pursuer_y.^2 + data.eci.pursuer_z.^2);
    evader_alt = (evader_radius - const.R_EARTH) / 1000;  % km
    pursuer_alt = (pursuer_radius - const.R_EARTH) / 1000; % km
    
    plot(data.eci.t/60, evader_alt, 'Color', plot_params.evader_color, 'LineWidth', 2);
    plot(data.eci.t/60, pursuer_alt, 'Color', plot_params.pursuer_color, 'LineWidth', 2);
    
    xlabel('Time [min]'); ylabel('Altitude [km]');
    title('Orbital Altitude Profiles');
    legend('Evader', 'Pursuer', 'Location', 'best');
    ylim([min([evader_alt; pursuer_alt])*0.95, max([evader_alt; pursuer_alt])*1.05]);
    
    % --- Subplot 3: Inter-spacecraft Distance ---
    ax3 = subplot(2, 3, 3);
    hold on; grid on; box on;
    
    inter_dist = sqrt((data.eci.evader_x - data.eci.pursuer_x).^2 + ...
                     (data.eci.evader_y - data.eci.pursuer_y).^2 + ...
                     (data.eci.evader_z - data.eci.pursuer_z).^2);
    
    plot(data.eci.t/60, inter_dist/1000, 'b-', 'LineWidth', 2);
    plot(data.eci.t/60, const.CAPTURE_RADIUS/1000 * ones(size(data.eci.t)), ...
         'r--', 'LineWidth', 1.5);
    plot(data.eci.t/60, const.EVASION_RADIUS/1000 * ones(size(data.eci.t)), ...
         'g--', 'LineWidth', 1.5);
    
    xlabel('Time [min]'); ylabel('Distance [km]');
    title('Inter-Spacecraft Distance (ECI)');
    legend('Distance', 'Capture', 'Evasion', 'Location', 'best');
    
    % --- Subplot 5: Orbital Elements Estimation ---
    ax5 = subplot(2, 3, 5);
    hold on; grid on; box on;
    
    % Calculate orbital periods (simplified)
    evader_period = 2*pi*sqrt(mean(evader_radius).^3/const.MU_EARTH) / 60; % minutes
    pursuer_period = 2*pi*sqrt(mean(pursuer_radius).^3/const.MU_EARTH) / 60;
    
    % Calculate eccentricity (simplified - assuming circular)
    evader_ecc = std(evader_radius) / mean(evader_radius);
    pursuer_ecc = std(pursuer_radius) / mean(pursuer_radius);
    
    orbital_data = [evader_period, pursuer_period;
                    mean(evader_alt), mean(pursuer_alt);
                    evader_ecc, pursuer_ecc];
    
    b = bar(orbital_data');
    b(1).FaceColor = plot_params.evader_color;
    b(2).FaceColor = plot_params.pursuer_color;
    set(gca, 'XTickLabel', {'Period [min]', 'Mean Alt [km]', 'Eccentricity'});
    ylabel('Value');
    title('Orbital Parameters Comparison');
    legend('Evader', 'Pursuer', 'Location', 'best');
    
    % --- Subplot 6: Ground Track (simplified) ---
    ax6 = subplot(2, 3, 6);
    hold on; grid on; box on;
    
    % Convert to latitude/longitude (simplified)
    evader_lat = asin(data.eci.evader_z ./ evader_radius) * 180/pi;
    evader_lon = atan2(data.eci.evader_y, data.eci.evader_x) * 180/pi;
    pursuer_lat = asin(data.eci.pursuer_z ./ pursuer_radius) * 180/pi;
    pursuer_lon = atan2(data.eci.pursuer_y, data.eci.pursuer_x) * 180/pi;
    
    plot(evader_lon, evader_lat, 'Color', plot_params.evader_color, 'LineWidth', 1.5);
    plot(pursuer_lon, pursuer_lat, 'Color', plot_params.pursuer_color, 'LineWidth', 1.5);
    
    xlabel('Longitude [deg]'); ylabel('Latitude [deg]');
    title('Ground Track Projection');
    legend('Evader', 'Pursuer', 'Location', 'best');
    xlim([-180, 180]); ylim([-90, 90]);
    grid on;
end

%% ========================================================================
%  FIGURE 3: DELTA-V ANALYSIS AND PROPULSION METRICS
% =========================================================================

if has_deltav
    fig3 = figure('Name', 'Delta-V and Propulsion Analysis', ...
                  'Position', [150, 100, 1400, 900], ...
                  'Color', 'white');
    
    % --- Subplot 1: Instantaneous Delta-V Magnitude ---
    ax1 = subplot(3, 3, 1);
    hold on; grid on; box on;
    
    plot(data.deltav.time, data.deltav.evader_dv_norm*1000, ...
         'Color', plot_params.evader_color, 'LineWidth', 1.5);
    plot(data.deltav.time, data.deltav.pursuer_dv_norm*1000, ...
         'Color', plot_params.pursuer_color, 'LineWidth', 1.5);
    
    xlabel('Time [s]'); ylabel('|ΔV| [mm/s]');
    title('Instantaneous Delta-V Magnitude');
    legend('Evader', 'Pursuer', 'Location', 'best');
    
    % --- Subplot 2: Cumulative Delta-V ---
    ax2 = subplot(3, 3, 2);
    hold on; grid on; box on;
    
    plot(data.deltav.time, data.deltav.evader_cumulative, ...
         'Color', plot_params.evader_color, 'LineWidth', 2);
    plot(data.deltav.time, data.deltav.pursuer_cumulative, ...
         'Color', plot_params.pursuer_color, 'LineWidth', 2);
    
    xlabel('Time [s]'); ylabel('Cumulative ΔV [m/s]');
    title('Total Delta-V Consumption');
    legend('Evader', 'Pursuer', 'Location', 'best');
    
    % --- Subplot 3: Delta-V Efficiency ---
    ax3 = subplot(3, 3, 3);
    hold on; grid on; box on;
    
    efficiency_ratio = data.deltav.pursuer_cumulative ./ (data.deltav.evader_cumulative + eps);
    plot(data.deltav.time, efficiency_ratio, 'k-', 'LineWidth', 2);
    plot(data.deltav.time, ones(size(data.deltav.time)), 'b--', 'LineWidth', 1.5);
    
    xlabel('Time [s]'); ylabel('Ratio P/E');
    title('Delta-V Efficiency Ratio');
    legend('Pursuer/Evader', 'Unity', 'Location', 'best');
    ylim([0, max(efficiency_ratio)*1.1]);
    
    % --- Subplots 4-5: Component Analysis ---
    if size(data.deltav.evader_dv, 2) == 3
        ax4 = subplot(3, 3, 4);
        hold on; grid on; box on;
        plot(data.deltav.time, data.deltav.evader_dv(:,1)*1000, 'r-', 'LineWidth', 1);
        plot(data.deltav.time, data.deltav.evader_dv(:,2)*1000, 'g-', 'LineWidth', 1);
        plot(data.deltav.time, data.deltav.evader_dv(:,3)*1000, 'b-', 'LineWidth', 1);
        xlabel('Time [s]'); ylabel('ΔV [mm/s]');
        title('Evader ΔV Components');
        legend('X', 'Y', 'Z', 'Location', 'best');
        
        ax5 = subplot(3, 3, 5);
        hold on; grid on; box on;
        plot(data.deltav.time, data.deltav.pursuer_dv(:,1)*1000, 'r-', 'LineWidth', 1);
        plot(data.deltav.time, data.deltav.pursuer_dv(:,2)*1000, 'g-', 'LineWidth', 1);
        plot(data.deltav.time, data.deltav.pursuer_dv(:,3)*1000, 'b-', 'LineWidth', 1);
        xlabel('Time [s]'); ylabel('ΔV [mm/s]');
        title('Pursuer ΔV Components');
        legend('X', 'Y', 'Z', 'Location', 'best');
    end
    
    % --- Subplot 6: Delta-V Direction in 3D ---
    ax6 = subplot(3, 3, 6);
    if size(data.deltav.evader_dv, 2) == 3
        hold on; grid on; box on;
        
        % Sample points for visualization
        sample_idx = 1:max(1, floor(n_points/20)):n_points;
        
        quiver3(zeros(size(sample_idx)), zeros(size(sample_idx)), zeros(size(sample_idx)), ...
               data.deltav.evader_dv(sample_idx,1), ...
               data.deltav.evader_dv(sample_idx,2), ...
               data.deltav.evader_dv(sample_idx,3), ...
               'Color', plot_params.evader_color, 'LineWidth', 1.5);
        quiver3(zeros(size(sample_idx)), zeros(size(sample_idx)), zeros(size(sample_idx)), ...
               data.deltav.pursuer_dv(sample_idx,1), ...
               data.deltav.pursuer_dv(sample_idx,2), ...
               data.deltav.pursuer_dv(sample_idx,3), ...
               'Color', plot_params.pursuer_color, 'LineWidth', 1.5);
        
        xlabel('ΔV_x'); ylabel('ΔV_y'); zlabel('ΔV_z');
        title('Delta-V Vector Directions');
        legend('Evader', 'Pursuer', 'Location', 'best');
        view(45, 25);
        axis equal;
    end
    
    % --- Subplot 7: Histogram Distribution ---
    ax7 = subplot(3, 3, 7);
    hold on;
    histogram(data.deltav.evader_dv_norm*1000, 30, ...
             'FaceColor', plot_params.evader_color, 'FaceAlpha', 0.6, ...
             'EdgeColor', 'none');
    histogram(data.deltav.pursuer_dv_norm*1000, 30, ...
             'FaceColor', plot_params.pursuer_color, 'FaceAlpha', 0.6, ...
             'EdgeColor', 'none');
    xlabel('|ΔV| [mm/s]'); ylabel('Frequency');
    title('Delta-V Magnitude Distribution');
    legend('Evader', 'Pursuer', 'Location', 'best');
    
    % --- Subplot 8: Box Plot Comparison ---
    ax8 = subplot(3, 3, 8);
    deltav_data_box = [data.deltav.evader_dv_norm*1000, data.deltav.pursuer_dv_norm*1000];
    boxplot(deltav_data_box, {'Evader', 'Pursuer'});
    ylabel('|ΔV| [mm/s]');
    title('Delta-V Statistical Comparison');
    grid on;
    
    % --- Subplot 9: Moving Average ---
    ax9 = subplot(3, 3, 9);
    window_size = max(1, floor(n_points/20));
    evader_smooth = movmean(data.deltav.evader_dv_norm, window_size);
    pursuer_smooth = movmean(data.deltav.pursuer_dv_norm, window_size);
    
    hold on; grid on; box on;
    plot(data.deltav.time, evader_smooth*1000, ...
         'Color', plot_params.evader_color, 'LineWidth', 2);
    plot(data.deltav.time, pursuer_smooth*1000, ...
         'Color', plot_params.pursuer_color, 'LineWidth', 2);
    xlabel('Time [s]'); ylabel('Smoothed |ΔV| [mm/s]');
    title(sprintf('Moving Average (Window: %d)', window_size));
    legend('Evader', 'Pursuer', 'Location', 'best');
end

%% ========================================================================
%  FIGURE 4: COMPREHENSIVE MISSION STATISTICS
% =========================================================================

fig4 = figure('Name', 'Mission Statistics and Summary', ...
              'Position', [200, 125, 1200, 800], ...
              'Color', 'white');

% --- Subplot 1: Performance Metrics Bar Chart ---
ax1 = subplot(2, 3, 1);

metrics_data = [
    rel_distance(end)/1000, const.CAPTURE_RADIUS/1000, const.EVASION_RADIUS/1000;
    data.deltav.evader_cumulative(end), data.deltav.pursuer_cumulative(end), 0;
    mean(rel_velocity), max(rel_velocity), min(rel_velocity);
];

b = bar(metrics_data);
b(1).FaceColor = [0.2 0.4 0.8];
b(2).FaceColor = [0.8 0.4 0.2];
b(3).FaceColor = [0.4 0.8 0.2];

set(gca, 'XTickLabel', {'Distance [km]', 'Total ΔV [m/s]', 'Velocity [m/s]'});
ylabel('Value');
title('Key Performance Metrics');
legend('Final/Evader', 'Capture/Pursuer', 'Evasion/Stats', 'Location', 'best');
grid on;

% --- Subplot 2: Time History Statistics ---
ax2 = subplot(2, 3, 2);

time_segments = 5;
segment_size = floor(n_points/time_segments);
segment_metrics = zeros(time_segments, 3);

for i = 1:time_segments
    idx_start = (i-1)*segment_size + 1;
    idx_end = min(i*segment_size, n_points);
    segment_metrics(i, 1) = mean(rel_distance(idx_start:idx_end));
    if has_deltav
        segment_metrics(i, 2) = sum(data.deltav.evader_dv_norm(idx_start:idx_end));
        segment_metrics(i, 3) = sum(data.deltav.pursuer_dv_norm(idx_start:idx_end));
    end
end

plot(1:time_segments, segment_metrics(:,1)/1000, 'b-o', 'LineWidth', 2);
hold on;
if has_deltav
    yyaxis right;
    plot(1:time_segments, segment_metrics(:,2)*1000, 'g-s', 'LineWidth', 1.5);
    plot(1:time_segments, segment_metrics(:,3)*1000, 'r-^', 'LineWidth', 1.5);
    ylabel('ΔV per Segment [mm/s]');
end
xlabel('Time Segment');
yyaxis left;
ylabel('Mean Distance [km]');
title('Temporal Evolution Analysis');
legend('Distance', 'Evader ΔV', 'Pursuer ΔV', 'Location', 'best');
grid on;

% --- Subplot 3: Polar Plot of Final Position ---
ax3 = subplot(2, 3, 3);
polarplot(atan2(data.trajectory.y(end), data.trajectory.x(end)), ...
         rel_distance(end), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
hold on;
theta = linspace(0, 2*pi, 100);
polarplot(theta, const.CAPTURE_RADIUS * ones(size(theta)), 'r--', 'LineWidth', 1.5);
polarplot(theta, const.EVASION_RADIUS * ones(size(theta)), 'g--', 'LineWidth', 1.5);
title('Final Position (Polar View)');
legend('Final Position', 'Capture', 'Evasion', 'Location', 'best');

% --- Subplot 4: Mission Timeline ---
ax4 = subplot(2, 3, 4);
hold on; grid on; box on;

% Create timeline events
events = [];
event_times = [];
event_labels = {};

% Find key events
[min_dist, min_idx] = min(rel_distance);
event_times = [event_times, min_idx];
event_labels{end+1} = sprintf('Closest: %.1f km', min_dist/1000);

[max_vel, max_vel_idx] = max(rel_velocity);
event_times = [event_times, max_vel_idx];
event_labels{end+1} = sprintf('Max Vel: %.1f m/s', max_vel);

if has_deltav
    [max_dv_e, max_dv_e_idx] = max(data.deltav.evader_dv_norm);
    event_times = [event_times, max_dv_e_idx];
    event_labels{end+1} = sprintf('Peak ΔV_E: %.1f mm/s', max_dv_e*1000);
end

stem(event_times, ones(size(event_times)), 'filled', 'LineWidth', 2);
text(event_times, ones(size(event_times))*1.1, event_labels, ...
     'Rotation', 45, 'FontSize', 9);
xlim([1, n_points]);
ylim([0, 2]);
xlabel('Time Step');
title('Mission Event Timeline');
set(gca, 'YTick', []);

% --- Subplot 5: Summary Text Panel ---
ax5 = subplot(2, 3, 5);
axis off;

summary_text = sprintf([
    '══════════════════════════════\n'
    '     MISSION SUMMARY REPORT    \n'
    '══════════════════════════════\n\n'
    'Scenario ID: %s\n'
    'Data Points: %d\n'
    'Mission Duration: %.1f min\n\n'
    '─── FINAL STATUS ───\n'
    'Outcome: %s\n'
    'Final Distance: %.2f km\n'
    'Final Velocity: %.2f m/s\n\n'
    '─── TRAJECTORY STATS ───\n'
    'Min Distance: %.2f km\n'
    'Max Distance: %.2f km\n'
    'Mean Distance: %.2f km\n'], ...
    config.scenario_id, n_points, ...
    data.deltav.time(end)/60, ...
    mission_outcome, ...
    rel_distance(end)/1000, ...
    rel_velocity(end), ...
    min(rel_distance)/1000, ...
    max(rel_distance)/1000, ...
    mean(rel_distance)/1000);

if has_deltav
    summary_text = [summary_text, sprintf([
        '\n─── PROPULSION STATS ───\n'
        'Evader Total ΔV: %.3f m/s\n'
        'Pursuer Total ΔV: %.3f m/s\n'
        'ΔV Ratio (P/E): %.2f\n'
        'Evader Mean ΔV: %.2f mm/s\n'
        'Pursuer Mean ΔV: %.2f mm/s\n'], ...
        data.deltav.evader_cumulative(end), ...
        data.deltav.pursuer_cumulative(end), ...
        data.deltav.pursuer_cumulative(end)/(data.deltav.evader_cumulative(end)+eps), ...
        mean(data.deltav.evader_dv_norm)*1000, ...
        mean(data.deltav.pursuer_dv_norm)*1000)];
end

if has_eci
    summary_text = [summary_text, sprintf([
        '\n─── ORBITAL STATS ───\n'
        'Mean Altitude: %.1f km\n'
        'Altitude Range: %.1f km\n'], ...
        mean([evader_alt; pursuer_alt]), ...
        max([evader_alt; pursuer_alt]) - min([evader_alt; pursuer_alt]))];
end

text(0.05, 0.95, summary_text, 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'FontName', 'FixedWidth', ...
     'FontSize', 10, 'BackgroundColor', [0.95 0.95 0.95]);

% --- Subplot 6: Success Criteria Evaluation ---
ax6 = subplot(2, 3, 6);

criteria = {'Distance < Capture', 'Distance > Evasion', 'Velocity < 10 m/s', ...
           'Duration < 20 min', 'ΔV < 10 m/s'};
criteria_met = [rel_distance(end) < const.CAPTURE_RADIUS, ...
               rel_distance(end) > const.EVASION_RADIUS, ...
               rel_velocity(end) < 10, ...
               data.deltav.time(end)/60 < 20, ...
               data.deltav.evader_cumulative(end) < 10];

colors_criteria = zeros(length(criteria), 3);
for i = 1:length(criteria)
    if criteria_met(i)
        colors_criteria(i, :) = [0 0.8 0];
    else
        colors_criteria(i, :) = [0.8 0 0];
    end
end

barh(criteria_met, 'FaceColor', 'flat', 'CData', colors_criteria);
set(gca, 'YTickLabel', criteria);
xlabel('Criteria Met (1) / Not Met (0)');
title('Mission Success Criteria');
xlim([0, 1.2]);
grid on;

%% ========================================================================
%  SAVE FIGURES
% =========================================================================

if config.save_figures
    fprintf('\nSaving figures...\n');
    figs = [fig1];
    if exist('fig2', 'var'), figs = [figs, fig2]; end
    if exist('fig3', 'var'), figs = [figs, fig3]; end
    if exist('fig4', 'var'), figs = [figs, fig4]; end
    
    for i = 1:length(figs)
        for fmt = config.save_format
            filename = fullfile(config.base_path, ...
                               sprintf('%s_figure_%d.%s', config.scenario_id, i, fmt{1}));
            switch fmt{1}
                case 'png'
                    print(figs(i), filename, '-dpng', '-r300');
                case 'fig'
                    savefig(figs(i), filename);
                case 'pdf'
                    print(figs(i), filename, '-dpdf', '-fillpage');
            end
        end
    end
    fprintf('✓ Figures saved to %s\n', config.base_path);
end

%% ========================================================================
%  ANIMATION SECTION (OPTIONAL)
% =========================================================================

if config.animation.enable
    fprintf('\nGenerating animation...\n');
    
    fig_anim = figure('Name', 'Trajectory Animation', ...
                     'Position', [300, 150, 1000, 800], ...
                     'Color', 'white');
    
    % Setup axes
    ax_anim = axes;
    hold on; grid on; box on;
    axis equal;
    
    % Set limits
    xlim([min(data.trajectory.x)-1000, max(data.trajectory.x)+1000]);
    ylim([min(data.trajectory.y)-1000, max(data.trajectory.y)+1000]);
    zlim([min(data.trajectory.z)-1000, max(data.trajectory.z)+1000]);
    
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    title('Animated Trajectory');
    view(45, 25);
    
    % Static elements
    plot3(0, 0, 0, 'o', 'MarkerSize', 12, ...
          'MarkerFaceColor', plot_params.evader_color, ...
          'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    
    % Capture/Evasion zones
    [xs, ys, zs] = sphere(20);
    surf(xs*const.CAPTURE_RADIUS, ys*const.CAPTURE_RADIUS, zs*const.CAPTURE_RADIUS, ...
         'FaceColor', plot_params.pursuer_color, 'EdgeColor', 'none', ...
         'FaceAlpha', 0.1);
    surf(xs*const.EVASION_RADIUS, ys*const.EVASION_RADIUS, zs*const.EVASION_RADIUS, ...
         'FaceColor', plot_params.evader_color, 'EdgeColor', 'none', ...
         'FaceAlpha', 0.05);
    
    % Initialize animated elements
    h_line = plot3(nan, nan, nan, 'b-', 'LineWidth', 2);
    h_point = plot3(nan, nan, nan, 'ro', 'MarkerSize', 10, ...
                   'MarkerFaceColor', 'r');
    h_text = text(0, 0, 0, '', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Animation parameters
    skip = max(1, floor(n_points/100));  % Limit to 100 frames
    frames = 1:skip:n_points;
    
    % Prepare video writer if saving
    if config.animation.save
        video_file = fullfile(config.base_path, ...
                             sprintf('%s_animation.mp4', config.scenario_id));
        v = VideoWriter(video_file, 'MPEG-4');
        v.FrameRate = config.animation.fps;
        open(v);
    end
    
    % Animation loop
    for k = frames
        % Update trajectory line
        set(h_line, 'XData', data.trajectory.x(1:k), ...
                   'YData', data.trajectory.y(1:k), ...
                   'ZData', data.trajectory.z(1:k));
        
        % Update current position
        set(h_point, 'XData', data.trajectory.x(k), ...
                    'YData', data.trajectory.y(k), ...
                    'ZData', data.trajectory.z(k));
        
        % Update text
        info_text = sprintf('Step: %d/%d\nDist: %.1f km', ...
                           k, n_points, rel_distance(k)/1000);
        set(h_text, 'Position', [data.trajectory.x(k), ...
                                 data.trajectory.y(k), ...
                                 data.trajectory.z(k) + 500], ...
                   'String', info_text);
        
        % Update title
        title(sprintf('Animated Trajectory (t = %.1f s)', ...
                     data.deltav.time(k)));
        
        drawnow;
        
        % Capture frame if saving
        if config.animation.save
            frame = getframe(fig_anim);
            writeVideo(v, frame);
        else
            pause(0.01);  % Small pause for visualization
        end
    end
    
    if config.animation.save
        close(v);
        fprintf('✓ Animation saved to %s\n', video_file);
    end
end

%% ========================================================================
%  EXPORT DATA ANALYSIS RESULTS
% =========================================================================

% Create analysis results structure
analysis_results = struct();
analysis_results.scenario_id = config.scenario_id;
analysis_results.mission_outcome = mission_outcome;
analysis_results.final_distance_km = rel_distance(end)/1000;
analysis_results.min_distance_km = min(rel_distance)/1000;
analysis_results.max_distance_km = max(rel_distance)/1000;
analysis_results.mean_distance_km = mean(rel_distance)/1000;

if has_deltav
    analysis_results.evader_total_deltav_ms = data.deltav.evader_cumulative(end);
    analysis_results.pursuer_total_deltav_ms = data.deltav.pursuer_cumulative(end);
    analysis_results.deltav_ratio = data.deltav.pursuer_cumulative(end) / ...
                                   (data.deltav.evader_cumulative(end) + eps);
end

if has_eci
    analysis_results.mean_altitude_km = mean([evader_alt; pursuer_alt]);
end

% Save analysis results
results_file = fullfile(config.base_path, ...
                       sprintf('%s_analysis_results.mat', config.scenario_id));
save(results_file, 'analysis_results');

% Display completion message
fprintf('\n════════════════════════════════════════\n');
fprintf('   Analysis Complete for %s\n', config.scenario_id);
fprintf('   Outcome: %s\n', mission_outcome);
fprintf('   Final Distance: %.2f km\n', rel_distance(end)/1000);
if has_deltav
    fprintf('   Total ΔV: E=%.3f m/s, P=%.3f m/s\n', ...
           data.deltav.evader_cumulative(end), ...
           data.deltav.pursuer_cumulative(end));
end
fprintf('════════════════════════════════════════\n\n');

% End of script