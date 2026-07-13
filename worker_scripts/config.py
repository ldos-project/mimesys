# Shared config for worker setup (install_remote_dependencies.py) and
# active-learning data collection (mimesys/collection/collect_training_data.py).
# Both read HOSTNAMES, USERNAME, PRIVATE_KEY_PATH from here.

# SSH connection settings
PORT = 22
USERNAME = "<FILLME>"
PRIVATE_KEY_PATH = "~/.ssh/id_rsa"

# Local paths
LOCAL_HOME_DIR = "/home/<FILLME>"

# Remote paths
REMOTE_HOME_DIR = "/users/<FILLME>"

# Controller hostname/IP that workers scp results back to. Must be reachable
# from each worker (so don't use "localhost" — use the public DNS the workers
# can resolve, or an IP).
MY_HOSTNAME = "<FILLME>"

# Hosts to initialize (install_remote_dependencies.py) and to profile against
# during active-learning rounds (collect_training_data.py).
HOSTNAMES = [
]
