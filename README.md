# PUMA 560 Manipulator Kinematics & Trajectory Planning

This repository contains a complete kinematic and trajectory-planning pipeline for a PUMA 560 robotic manipulator.  
The project implements the core mathematical models and algorithms required to describe, control, and simulate a 6-DOF industrial robot arm.

## Overview

The work includes:

- Construction of the Denavit-Hartenberg (DH) parameter table for the PUMA 560 architecture
- Forward kinematics
- Differential (velocity) kinematics
- Inverse kinematics
- Trajectory planning for multiple paths (without obstacle avoidance)

## Features

### 1. DH Parameterization

A complete DH table is implemented following the standard PUMA 560 configuration.  
This allows automatic generation of transformation matrices for each joint.

### 2. Forward Kinematics

- Homogeneous transformation matrices derived from DH parameters
- End-effector pose computation (position and orientation)

### 3. Velocity (Differential) Kinematics

- Jacobian matrix formulation
- Mapping between joint velocities and Cartesian velocities
- Singularities considered in analytical Jacobian computation

### 4. Inverse Kinematics

- Closed-form IK for the PUMA 560 geometry
- Multiple solution branches where applicable
- Validation of solutions through forward-kinematics consistency checks

### 5. Trajectory Planning

Implemented trajectory planning for various paths, including:

- Joint-space point-to-point trajectories
- Multi-segment trajectories
- Smooth motion profiles using common interpolation schemes (e.g., cubic or quintic polynomials)
- End-effector tracking in free space (no obstacles)
