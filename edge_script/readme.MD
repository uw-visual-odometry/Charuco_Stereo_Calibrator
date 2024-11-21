# Raspberry Pi Snapshot & Downloader

This repository contains scripts to capture images from Raspberry Pi cameras and download them to a local machine.

## Files

1. **`rpi_snapshot.bash`**  
   Runs on Raspberry Pi to capture images using `libcamera-still`.  
   - Captures images from two cameras (`camera 0` and `camera 1`).  
   - Saves images with `<epoch>_left.jpg` and `<epoch>_right.jpg`.

2. **`snapshot.py`**  
   Runs on the local machine to:  
   - Trigger the image capture on the Raspberry Pi via SSH.  
   - Download captured images using SCP.  

---

## Requirements

### Raspberry Pi  
- `libcamera` installed (`sudo apt install libcamera-apps`).  
- SSH enabled and accessible.
- It's in the same network through WIFI or LAN

### Local Machine  
- Python 3+ with `paramiko` and `scp` (`pip install paramiko scp`).  

---

## Usage

### On Raspberry Pi  
1. Place `rpi_snapshot.bash` at `/home/pi/snapshot_input.bash`.  
2. Make it executable:  
   ```bash
   chmod +x /home/pi/snapshot_input.bash
   ```
3. Run manually:  
   ```bash
   bash /home/pi/snapshot_input.bash <epoch_time>
   ```

### On Local Machine  
1. Update `hostname`, `username`, and `password` in `snapshot.py`.  
2. Run the script:  
   ```bash
   python snapshot.py
   ```

---

## Outputs

- **Raspberry Pi**: Captured images are saved in `left/` and `right/` directories.  
- **Local Machine**: Images are downloaded to `downloaded_images/left` and `downloaded_images/right`.  