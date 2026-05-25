# Shared config for worker setup (install_remote_dependencies.py) and
# active-learning data collection (mimesys/collection/collect_training_data.py).
# Both read HOSTNAMES, USERNAME, PRIVATE_KEY_PATH from here.

# SSH connection settings
PORT = 22
USERNAME = "dhkim"
PRIVATE_KEY_PATH = "/home/dhkim/.ssh/id_rsa_utns"

# Local paths
LOCAL_HOME_DIR = "/home/dhkim"

# Remote paths
REMOTE_HOME_DIR = "/users/dhkim"

# Controller hostname/IP that workers scp results back to. Must be reachable
# from each worker (so don't use "localhost" — use the public DNS the workers
# can resolve, or an IP).
MY_HOSTNAME = "mew3"

# Hosts to initialize (install_remote_dependencies.py) and to profile against
# during active-learning rounds (collect_training_data.py).
HOSTNAMES = [
    "c220g5-110923.wisc.cloudlab.us",
    "c220g5-111203.wisc.cloudlab.us",
    "c220g5-111205.wisc.cloudlab.us",
    "c220g5-111212.wisc.cloudlab.us",
    "c220g5-111204.wisc.cloudlab.us",
    "c220g5-111224.wisc.cloudlab.us",
    # "c220g5-111214.wisc.cloudlab.us",  # excluded: hpcperfstatsd IO counter is dead (returns 0 for all workloads incl. Hdd_1MB) — re-add after fixing
    "c220g5-111213.wisc.cloudlab.us",
    "c220g5-111219.wisc.cloudlab.us",
    "c220g5-111227.wisc.cloudlab.us",
    "c220g5-111220.wisc.cloudlab.us",
    "c220g5-111223.wisc.cloudlab.us",
    "c220g5-111210.wisc.cloudlab.us",
    "c220g5-110501.wisc.cloudlab.us",
    "c220g5-111230.wisc.cloudlab.us",
    "c220g5-111217.wisc.cloudlab.us",
]

# Plans-per-machine per active-learning round. Round size is
# PER_MACHINE_BATCH * len(HOSTNAMES) — e.g. 16 × 4 hosts = 64 plans/round.
PER_MACHINE_BATCH = 16
