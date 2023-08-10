import platform
import sysconfig

def get_arch():
    p = platform.platform() + '-' + sysconfig.get_platform()
    for arch in [ 'arm64', 'aarch64', 'armv7', 'arm', ]:
        if arch in p:
            return arch
    return None
