import os
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: launch_rank.py <script.py> [args...]')

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(',')
                   if device.strip()]
        if local_rank >= len(devices):
            raise RuntimeError(
                f'LOCAL_RANK={local_rank} is out of range for '
                f'CUDA_VISIBLE_DEVICES={visible_devices}')
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[local_rank]
        os.environ['UNIAV_ORIGINAL_LOCAL_RANK'] = str(local_rank)
        os.environ['LOCAL_RANK'] = '0'

    script = sys.argv[1]
    args = [sys.executable, script, *sys.argv[2:]]
    os.execv(sys.executable, args)


if __name__ == '__main__':
    main()
