import paramiko
from scp import SCPClient, SCPException
import os
import time


def create_ssh_client(server, port, user, password):
    try:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, username=user, password=password, timeout=10)
        return client
    except paramiko.AuthenticationException:
        raise Exception("Authentication failed while establishing SSH connection.")
    except paramiko.SSHException as ssh_error:
        raise Exception(f"SSH connection error: {ssh_error}")
    except Exception as e:
        raise Exception(f"Failed to create SSH client: {e}")


def execute_command(ssh_client, command):
    try:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        errors = stderr.read().decode().strip()
        if errors:
            raise Exception(f"Error executing command `{command}`: {errors}")
        return stdout.read().decode().strip()
    except paramiko.SSHException as ssh_error:
        raise Exception(f"Failed to execute command `{command}`: {ssh_error}")
    except Exception as e:
        raise Exception(f"Unexpected error during command execution `{command}`: {e}")


def download_image(ssh_client, remote_file, local_dir):
    try:
        if not remote_file:
            raise ValueError("Remote file path is empty or None.")

        print(f"Downloading {remote_file} to {local_dir}")
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.get(remote_file, local_path=local_dir)

    except SCPException as scp_error:
        raise Exception(f"SCP error while downloading {remote_file}: {scp_error}")
    except FileNotFoundError:
        raise Exception(f"Local directory {local_dir} does not exist.")
    except Exception as e:
        raise Exception(f"Unexpected error while downloading {remote_file}: {e}")


def main():
    # SSH connection details - Update these as needed
    hostname = "192.168.1.1"

    port = 22  # Default SSH port
    username = "name"  # SSH username
    password = "pw"  # SSH password

    # Local directory where downloaded images will be saved
    base_local_dir = "./downloaded_images"
    os.makedirs(
        base_local_dir, exist_ok=True
    )  # Create the base directory if it doesn't already exist

    # Get the current epoch time to uniquely identify the files
    epoch = int(time.time())

    try:
        # Establish SSH connection
        print("Establishing SSH connection...")
        ssh_client = create_ssh_client(hostname, port, username, password)

        # Trigger the remote snapshot script with the current epoch time as an argument
        print("Executing snapshot script on the remote device...")
        command = f"bash /home/pi/snapshot_input.bash {epoch}"
        execute_command(ssh_client, command)

        # Wait for the remote script to process and generate files
        print("Waiting for snapshot script to complete...")
        time.sleep(5)  # Increase delay if the remote script takes more time

        # Loop through the 'left' and 'right' directories to download images
        for side in ["left", "right"]:
            remote_dir = f"/home/pi/{side}"  # Remote directory on the SSH server
            remote_file = f"{remote_dir}/{epoch}_{side}.jpg"  # Remote file path with timestamp and side

            # Create local subdirectory for the current 'side' (left or right)
            local_subdir = os.path.join(base_local_dir, side)
            os.makedirs(
                local_subdir, exist_ok=True
            )  # Create the subdir if it doesn't exist

            # Download the image file from the remote server
            print(f"Attempting to download {side} image...")
            download_image(ssh_client, remote_file, local_subdir)

        # Close the SSH connection once all downloads are complete
        ssh_client.close()
        print("All images downloaded successfully. Process complete.")

    except Exception as e:
        # General error handling with an error message
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
