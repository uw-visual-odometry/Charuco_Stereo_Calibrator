  GNU nano 7.2                              snapshot_input.bash                                       
#!/bin/bash

# Check if epoch argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <epoch>"
    exit 1
fi

epoch=$1  # Use the provided epoch

# Create directories if they do not exist
mkdir -p left
mkdir -p right

# Capture a snapshot from camera 0
if libcamera-still --camera 0 -o left/${epoch}_left.jpg; then
    echo "Snapshot taken from camera 0"
    
    # Capture a snapshot from camera 1
    if libcamera-still --camera 1 -o right/${epoch}_right.jpg; then
        echo "Snapshot taken from camera 1"
    else
        echo "Failed to capture snapshot from camera 1"
    fi
else
    echo "Failed to capture snapshot from camera 0"
fi
