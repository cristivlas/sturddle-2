import platform
import subprocess
import sysconfig

def get_arch():
    p = platform.platform() + '-' + sysconfig.get_platform()
    for arch in [ 'arm64', 'aarch64', 'armv7', 'arm', ]:
        if arch in p:
            return arch
    return None

def is_apple_silicon():
    if platform.system() == 'Darwin':
        # Using subprocess to execute 'uname -m' and capture the output
        arch = subprocess.check_output(['uname', '-a']).decode('utf-8')
        return '_ARM64_' in arch
    return False

