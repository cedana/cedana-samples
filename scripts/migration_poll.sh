#!/bin/bash

#!/bin/bash

# Check if job ID and instance IP are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <job-id> <instance-ip>"
    exit 1
fi

INSTANCE_A="ubuntu@216.86.162.76"
INSTANCE_IP="${INSTANCE_A#*@}" # Extract IP from user@host format

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
INSTANCE_IP=$2

# Function to check for checkpoint and restore
check_and_restore() {
    # Start the loading animation
    BLA::start_loading_animation "${BLA_football[@]}"

    while true; do
        while nc -z -w 2 "$INSTANCE_IP" 22; do
            sleep 0.1
        done

        BLA::stop_loading_animation
        cedana restore job "$JOB_ID" -a --tcp-close

        # Exit after restoring
        break

        # Sleep briefly to avoid hammering the CPU
        sleep 1
    done
}

# Run the function
check_and_restore
