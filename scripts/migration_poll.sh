#!/bin/bash

# Check if job ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <job-id>"
    exit 1
fi

# Load the bash_loading_animations library
# Replace /path/to/bash_loading_animations.sh with the actual path to the library
source bash_loading_animations.sh

# Function to clean up and stop the loading animation on script exit or interrupt
cleanup() {
    BLA::stop_loading_animation
    exit
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

JOB_ID=$1

# Function to check for checkpoint and restore
check_and_restore() {
    # Print the initial message and start the spinner

    # Start the loading animation
    BLA::start_loading_animation "${BLA_football[@]}"

    while true; do
        # Get the checkpoint list for the given job ID
        CHECKPOINT_LIST=$(cedana checkpoint list "$JOB_ID")

        # Check if the checkpoint list contains any checkpoint
        if echo "$CHECKPOINT_LIST" | grep -q "MiB"; then
            # Stop the loading animation
            BLA::stop_loading_animation

            # Extract the checkpoint ID from the list
            CHECKPOINT_ID=$(echo "$CHECKPOINT_LIST" | awk 'NR==2 {print $1}')

            # Restore the checkpoint
            echo -e "\nCheckpoint detected with ID: $CHECKPOINT_ID"
            echo "Restoring checkpoint..."
            cedana restore job "$JOB_ID" -a

            # Exit the loop after restoring
            break
        fi

        # Sleep briefly to avoid hammering the CPU
        sleep 1
    done
}

# Run the function
check_and_restore
