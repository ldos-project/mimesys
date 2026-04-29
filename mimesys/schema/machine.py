from __future__ import annotations

from dataclasses import dataclass
from mimesys.schema.constants import machine_types

import paramiko


@dataclass(frozen=True)
class Machine:
    hostname: str
    machine_type: str

    @classmethod
    def from_hostname(cls, hostname: str) -> Machine:
        machine_type = hostname.split("-")[0]
        if machine_type not in machine_types:
            raise ValueError(f"Unknown machine type: {machine_type}")
        return cls(hostname=hostname, machine_type=machine_type)

    def initialize_connection(
            self, username: str, private_key_path: str
    ) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
        print(f"Connecting to {self.hostname} as {username} using key {private_key_path}")

        key = paramiko.RSAKey.from_private_key_file(private_key_path)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.hostname, username=username, pkey=key)
        scp = paramiko.SFTPClient.from_transport(client.get_transport())
        return client, scp

    def close_connection(self, scp: paramiko.SFTPClient, client: paramiko.SSHClient) -> None:
        scp.close()
        client.close()

    def file_transfer(self, scp: paramiko.SFTPClient, file_path: str, destination: str) -> None:
        print(f"Transferring file {file_path} to {self.hostname}:{destination}")
        results = scp.put(file_path, destination)
        print(results)

    def run_command(self, client: paramiko.SSHClient, command: str) -> str:
        _, stdout, stderr = client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()  # Wait for command to complete
        if exit_status != 0:
            raise RuntimeError(f"Command failed with exit status {exit_status}: {stderr.read().decode()}")
        return stdout.read().decode()

    def run_command_background(self, client: paramiko.SSHClient, command: str) -> None:
        # Wrap in bash -c so shell builtins (cd) and operators (&&) work,
        # then nohup + setsid ensure the process survives SSH connection close.
        import shlex
        client.exec_command(f"nohup setsid bash -c {shlex.quote(command)} >/dev/null 2>&1 &")
