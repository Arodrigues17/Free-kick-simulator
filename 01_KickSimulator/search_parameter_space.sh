#!/bin/bash

# Adjust path to executable if needed
EXECUTABLE="./main"
OUTPUT_DIR="output"

# Ensure output directory exists
rm -rf "$OUTPUT_DIR"  # Clear previous output
mkdir -p "$OUTPUT_DIR"

# Define parameters
X_LINES=(20 30 40)  # Three x positions
Z_MIN=0             # Lower bound for z-line
Z_MAX=68            # Upper bound for z-line
NUM_Z_POINTS=10     # Points along each z-line

Y_START=0
ANGLE_Y_VALUES=(0 10 20 30)  # Vertical angles in degrees
DIRECTION_ANGLES=(-30 -20 -10 0 10 20 30)
SPINS=(-60 -50 -40 -30 -20 -10 0 10 20 30 40 50 60)
VELOCITIES=(15 22 30)  # weak, average, strong

# Compute step size for z (integer division)
Z_STEP=$(( (Z_MAX - Z_MIN) / (NUM_Z_POINTS - 1) ))

echo "Starting parameter search..."
LOG_FILE="$OUTPUT_DIR/parameter_log.txt"
> "$LOG_FILE"  # Clear log

TOTAL_COUNT=0
SUCCESS_COUNT=0

for X_POS in "${X_LINES[@]}"; do
  for j in $(seq 0 $((NUM_Z_POINTS - 1))); do
    Z_POS=$(( Z_MIN + j * Z_STEP ))

    for v0 in "${VELOCITIES[@]}"; do
      for spin_y in "${SPINS[@]}"; do
        for angle_y in "${ANGLE_Y_VALUES[@]}"; do
          for dir_angle in "${DIRECTION_ANGLES[@]}"; do

            # Total attempts counter
            TOTAL_COUNT=$((TOTAL_COUNT + 1))

            # Create unique output file name for successful cases
            OUTPUT_FILE="$OUTPUT_DIR/wavepoints_${SUCCESS_COUNT}.csv"

            # Call the C++ program
            $EXECUTABLE $v0 $angle_y $dir_angle 0 $spin_y 0 $X_POS $Y_START $Z_POS "$OUTPUT_FILE" >> /dev/null

            if [ $? -eq 0 ]; then
              echo "✅ Goal! Saved to $OUTPUT_FILE"
              echo "$OUTPUT_FILE: x0=$X_POS, z0=$Z_POS, v0=$v0, angle_y=$angle_y, dir_angle=$dir_angle, spin_y=$spin_y" >> "$LOG_FILE"
              SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
              rm -f "$OUTPUT_FILE"
            fi

          done
        done
      done
    done
  done
done

# Final summary
echo "Search complete!"
echo "Total simulations conducted: $TOTAL_COUNT" | tee -a "$LOG_FILE"
echo "Total successful goals: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
echo "Parameter log saved to $LOG_FILE"
