#!/bin/bash

# Batch script to render multiple trajectory configurations
# Usage: ./batch_render.sh

SCRIPT_NAME="render.py"

echo "Starting batch trajectory rendering..."
echo "Script: $SCRIPT_NAME"
echo "======================================"

# Create output directory
mkdir -p videos

# Configuration
FPS=30
WIDTH=1280
HEIGHT=720

# Define path types and interpolation types to test
PATH_TYPES=("linear" "arc" "circular" "parabolic" "rrt")
INTERP_TYPES=("cubic" "quintic" "lspb" "bangbang")

# Example 1: Render all path types with bangbang interpolation
echo ""
echo "Rendering all path types with bangbang interpolation..."
for path in "${PATH_TYPES[@]}"; do
    echo "  - Processing: path=$path, interp=bangbang"
    python "$SCRIPT_NAME" \
        --path-type $path \
        --interp-type bangbang \
        --output "videos/traj_${path}_bangbang.mp4" \
        --fps $FPS \
        --width $WIDTH \
        --height $HEIGHT \
        --no-display
done

# Example 2: Render all interpolation types with RRT path
echo ""
echo "Rendering all interpolation types with RRT path..."
for interp in "${INTERP_TYPES[@]}"; do
    echo "  - Processing: path=rrt, interp=$interp"
    python "$SCRIPT_NAME" \
        --path-type arc \
        --interp-type $interp \
        --output "videos/traj_rrt_${interp}.mp4" \
        --fps $FPS \
        --width $WIDTH \
        --height $HEIGHT \
        --no-display
done

# Example 3: Render specific combinations
echo ""
echo "Rendering specific combinations..."

# Linear path with different interpolations
python "$SCRIPT_NAME" -p linear -i cubic -o videos/linear_cubic.mp4 --fps $FPS --width $WIDTH --height $HEIGHT --no-display
python "$SCRIPT_NAME" -p linear -i lspb -o videos/linear_lspb.mp4 --fps $FPS --width $WIDTH --height $HEIGHT --no-display

# Circular path with different interpolations
python "$SCRIPT_NAME" -p circular -i quintic -o videos/circular_quintic.mp4 --fps $FPS --width $WIDTH --height $HEIGHT --no-display
python "$SCRIPT_NAME" -p circular -i bangbang -o videos/circular_bangbang.mp4 --fps $FPS --width $WIDTH --height $HEIGHT --no-display

echo ""
echo "======================================"
echo "✓ Batch rendering complete!"
echo "Videos saved in ./videos/ directory"
echo ""
echo "Generated videos:"
ls -lh videos/*.mp4