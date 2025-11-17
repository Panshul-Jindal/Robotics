%% ==========================================================
%   PUMA 560 – Cartesian Straight-Line Trajectory (IK)
% ==========================================================
clear; clc; close all;

%% --- Load PUMA560 CAD Robot from URDF ---
urdfPath = fullfile("puma560_description","urdf","puma560_robot.urdf");
puma = importrobot(urdfPath, "DataFormat","row");

eeName = "link7";   % from your URDF
fprintf("End-effector link: %s\n", eeName);

%% --- Show initial robot pose ---
figure('Color','white');
ax = axes('Parent', gcf);
show(puma, homeConfiguration(puma), "Parent",ax, "Visuals","on");
view(135,25); axis equal; grid on;

% Fix world size
xlim([-1.2 1.2]);
ylim([-1.2 1.2]);
zlim([0 1.8]);


% Make visuals cleaner


title("PUMA 560 – Cartesian Straight-Line Trajectory");
hold on; camlight;


%% ==========================================================
%     DEFINE CARTESIAN STRAIGHT-LINE WAYPOINTS
% ==========================================================

q0 = homeConfiguration(puma);

% Initial pose
T0 = getTransform(puma, q0, eeName);
p0 = T0(1:3,4)';

% Target point (modify if needed)
c =0.5
p1 = p0 + [c, -c,c];

N = 150;

cartesianPoints = [linspace(p0(1), p1(1), N)' ...
                   linspace(p0(2), p1(2), N)' ...
                   linspace(p0(3), p1(3), N)'];


%% ==========================================================
%          INVERSE KINEMATICS FOR EACH WAYPOINT
% ==========================================================

ik = inverseKinematics("RigidBodyTree", puma);
weights = [1 1 1 1 1 1];

R0 = tform2rotm(T0);

qSol = zeros(N, numel(q0));
qSol(1,:) = q0;

for k = 2:N
    Ttarget = eye(4);
    Ttarget(1:3,1:3) = R0;
    Ttarget(1:3,4)   = cartesianPoints(k,:)';

    % IMPORTANT FIX: no transpose, row vector only
    qSol(k,:) = ik(eeName, Ttarget, weights, qSol(k-1,:));
end


%% ==========================================================
%            SIMULATE MOTION + PLOT EE PATH
% ==========================================================

eePath = zeros(N,3);

for k = 1:N
    show(puma, qSol(k,:), ...
         "Parent",ax, ...
         "PreservePlot",false, ...
         "FastUpdate",true, ...
         "Visuals","on");

    T = getTransform(puma, qSol(k,:), eeName);
    eePath(k,:) = T(1:3,4)';

    plot3(eePath(1:k,1), eePath(1:k,2), eePath(1:k,3), 'r', "LineWidth",2);
    drawnow limitrate;
    pause(0.1);   % slows the animation
end

xlabel X; ylabel Y; zlabel Z;
title("PUMA 560 – Straight-Line Cartesian Trajectory Completed");
grid on;
