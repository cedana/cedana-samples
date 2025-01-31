#!/bin/bash

#!/bin/bash

# Check if job ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <job-id>"
    exit 1
fi

INSTANCE_A="ubuntu@216.86.162.76"
INSTANCE_IP="${INSTANCE_A#*@}"  # Extract IP from user@host format

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
    # Start the loading animation
    BLA::start_loading_animation "${BLA_football[@]}"

    while true; do
        # Get the checkpoint list for the given job ID
        CHECKPOINT_LIST=$(cedana checkpoint list "$JOB_ID")
        # Sync with DB
        cedana ps > /dev/null

        # Check if the checkpoint list contains any checkpoint
        if echo "$CHECKPOINT_LIST" | grep -q "dump"; then
            # Stop the loading animation
            BLA::stop_loading_animation

            # Extract the checkpoint ID from the list
            CHECKPOINT_ID=$(echo "$CHECKPOINT_LIST" | awk 'NR==2 {print $1}')

            # Wait for instance to die before restoring
            echo -e "\nCheckpoint detected with ID: $CHECKPOINT_ID"
            echo "Waiting for instance to become unreachable..."
            FILE="/root/shared-mount/dump-process-${JOB_ID}.tar"

            while [[ ! -f "$FILE" ]]; do
                sleep 1  # Wait for 1 second before checking again
            done

            cp -r "$FILE" "/root/dump-process-${JOB_ID}.tar"

            while nc -z -w 2 "$INSTANCE_IP" 22; do
                sleep 1
            done

            echo "Instance is unreachable. Restoring checkpoint..."
            cedana restore job "$JOB_ID" -a

            # Exit after restoring
            break
        fi

        # Sleep briefly to avoid hammering the CPU
        sleep 1
    done
}

# Run the function
check_and_restore
