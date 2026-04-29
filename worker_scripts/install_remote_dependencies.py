import paramiko
import os
import concurrent.futures

import config

key = paramiko.RSAKey.from_private_key_file(config.PRIVATE_KEY_PATH)

num_hosts = len(config.HOSTNAMES)
scripts_dir = os.path.dirname(os.path.abspath(__file__))

def process_host(hostname):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=hostname, port=config.PORT, username=config.USERNAME, pkey=key)
        scp = paramiko.SFTPClient.from_transport(client.get_transport())
        if hostname in config.HOSTNAMES:
            # Copy SSH keys to enable outbound git clones from the worker
            scp.put(f"{config.LOCAL_HOME_DIR}/.ssh/config", f"{config.REMOTE_HOME_DIR}/.ssh/config")
            for key_file in ["id_rsa", "id_rsa.pub"]:
                local = f"{config.LOCAL_HOME_DIR}/.ssh/{key_file}"
                remote = f"{config.REMOTE_HOME_DIR}/.ssh/{key_file}"
                scp.put(local, remote)
            client.exec_command(f"chmod 600 {config.REMOTE_HOME_DIR}/.ssh/id_rsa")

            # Copy worker setup scripts
            for script in ["install_executor.sh", "stats.x"]:
                scp.put(f"{scripts_dir}/{script}", f"{config.REMOTE_HOME_DIR}/{script}")

            # Copy profiling scripts
            for script in ["collect_mimesys_metrics.sh"]:
                scp.put(f"{scripts_dir}/{script}", f"{config.REMOTE_HOME_DIR}/{script}")

            # Clone repos and start install in a tmux session
            stdin, stdout, stderr = client.exec_command(
                f'cd {config.REMOTE_HOME_DIR} && git clone https://github.com/TACC/HPCPerfStats.git && cd HPCPerfStats && git checkout 2.4'
            )
            for line in stderr:
                print(line.strip('\n'))

            stdin, stdout, stderr = client.exec_command(
                f'tmux new-session -d -s install "bash {config.REMOTE_HOME_DIR}/install_executor.sh"'
            )
            for line in stderr:
                print(line.strip('\n'))
    except Exception as e:
        print(f"Error connecting or executing command on {hostname}: {e}")
    finally:
        client.close()


with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(process_host, hostname)
        for hostname in config.HOSTNAMES
    ]
    concurrent.futures.wait(futures)
