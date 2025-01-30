#!/bin/bash

# Check if job ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <job-id>"
    exit 1
fi

JOB_ID=$1

# Function to check for checkpoint and restore
check_and_restore() {
    while true; do
        # Get the checkpoint list for the given job ID
        CHECKPOINT_LIST=$(cedana checkpoint list "$JOB_ID")

        # Check if the checkpoint list contains any checkpoint
        if echo "$CHECKPOINT_LIST" | grep -q "MiB"; then
            # Extract the checkpoint ID from the list
            CHECKPOINT_ID=$(echo "$CHECKPOINT_LIST" | awk 'NR==2 {print $1}')

            # Restore the checkpoint
            echo "Checkpoint detected with ID: $CHECKPOINT_ID"
            echo "Restoring checkpoint..."
            cedana restore "$CHECKPOINT_ID"

            # Exit the loop after restoring
            break
        else
            echo "No checkpoint found for job ID: $JOB_ID. Retrying in 5 seconds..."
            sleep 5
        fi
    done
}

# Run the function
check_and_restore
