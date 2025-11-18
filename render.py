#!/usr/bin/env python3
"""
Batch script to render multiple trajectory configurations
Usage: python batch_render.py
"""

import subprocess
import os
from pathlib import Path
import sys
from datetime import datetime

def run_trajectory(path_type, interp_type, output_file, fps=30, headless=True):
    """Run a single trajectory simulation."""
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "simulate.py",
        "--path-type", path_type,
        "--interp-type", interp_type,
        "--output", output_file,
        "--fps", str(fps)
    ]
    
    if headless:
        cmd.append("--no-display")
    
    print(f"\n{'='*60}")
    print(f"Running: path={path_type}, interp={interp_type}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Success: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {output_file}")
        print(f"  Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)


def main():
    """Main batch rendering function."""
    
    # Create output directory
    output_dir = Path("videos")
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    path_types = ["linear", "arc", "parabolic", "rrt"]
    interp_types = ["cubic", "quintic", "lspb", "bangbang"]
    fps = 30
    headless = True  # Set to False to see the viewer window
    
    print(f"\n{'='*70}")
    print(f"BATCH TRAJECTORY RENDERING")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Headless mode: {headless}")
    print(f"Video FPS: {fps}")
    
    results = []
    
    # Strategy 1: All path types with one interpolation method
    print(f"\n{'#'*70}")
    print(f"# Strategy 1: All path types with bangbang interpolation")
    print(f"{'#'*70}")
    
    for path in path_types:
        output_file = output_dir / f"traj_{path}_bangbang.mp4"
        success = run_trajectory(path, "bangbang", str(output_file), fps, headless)
        results.append((path, "bangbang", success))
    
    # Strategy 2: All interpolation types with one path
    print(f"\n{'#'*70}")
    print(f"# Strategy 2: All interpolation types with RRT path")
    print(f"{'#'*70}")
    
    for interp in interp_types:
        output_file = output_dir / f"traj_arc_{interp}.mp4"
        success = run_trajectory("arc", interp, str(output_file), fps, headless)
        results.append(("arc", interp, success))

    # Strategy 3: Specific interesting combinations
    print(f"\n{'#'*70}")
    print(f"# Strategy 3: Specific combinations")
    print(f"{'#'*70}")
    
    combinations = [
        ("linear", "cubic"),
        ("linear", "lspb"),
        ("arc", "quintic"),
        ("parabolic", "bangbang"),
    ]
    
    for path, interp in combinations:
        output_file = output_dir / f"traj_{path}_{interp}.mp4"
        success = run_trajectory(path, interp, str(output_file), fps, headless)
        results.append((path, interp, success))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"BATCH RENDERING SUMMARY")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults:")
    
    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful
    
    for path, interp, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {path:12s} + {interp:10s}")
    
    print(f"\nTotal: {len(results)} renderings")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # List generated files
    print(f"\n{'='*70}")
    print(f"Generated video files:")
    print(f"{'='*70}")
    
    video_files = sorted(output_dir.glob("*.mp4"))
    if video_files:
        for vf in video_files:
            size_mb = vf.stat().st_size / (1024 * 1024)
            print(f"  {vf.name:40s} ({size_mb:.2f} MB)")
    else:
        print("  No video files found")
    
    print(f"\n✓ Batch rendering complete!")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())