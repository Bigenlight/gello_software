"""Policy-to-viewer run discovery. No ROS calls and no policy/robot control."""
import json
import os
import sys
import time
import hashlib
import shlex
from pathlib import Path

DEFAULT_HUB = Path('/home/junhyeong/gello_software_jazzy/ros2_ur_ws/log')

def hub_root():
    return Path(os.environ.get('REAL_EVAL_LOG_HUB', str(DEFAULT_HUB))).expanduser().resolve()

def episode_root(run):
    run = Path(run)
    manifest = json.loads((run/'launch_manifest.json').read_text())
    return Path(manifest.get('recording', {}).get('ros_episode_dir') or run/'episodes')

def episode_files(run):
    run = Path(run).resolve()
    root = episode_root(run)
    for file in root.glob('ep_*/episode.json'):
        try:
            meta = json.loads(file.read_text())
            owner = meta.get('metadata', {}).get('recording_run_dir')
            if owner == str(run) or (owner is None and root == run/'episodes'):
                yield file, meta
        except (OSError, ValueError, KeyError):
            continue

def prepare_shared(run, profile):
    """Group equal resolved commands/artifacts, while retaining private server logs."""
    run = Path(run).resolve()
    manifest_path = run/'launch_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    base = run.parent.parent if run.parent.name == '.sessions' else run.parent
    commands = {}
    for name, command in manifest['commands'].items():
        words = shlex.split(command)
        if name == 'server_shell' and '--log-dir' in words:
            index = words.index('--log-dir')
            words[index+1] = '<session-log-dir>'
        if name == 'adapter_shell' and '--prompt-file' in words:
            # VLA launchers pass a per-session pending_prompt.json; it is run
            # bookkeeping, not policy configuration, so it must not split runs.
            index = words.index('--prompt-file')
            words[index+1] = '<session-prompt-file>'
        commands[name] = words
    policy = json.loads(json.dumps(manifest['policy']))
    if isinstance(policy.get('prompt_control'), dict) and 'pending_file' in policy['prompt_control']:
        policy['prompt_control']['pending_file'] = '<session-prompt-file>'
    inputs = {k:v for k,v in manifest['artifacts']['inputs'].items()
              if k in ('checkpoint','flags','norm_stats','server')}
    yaml_hashes = {}
    for word in commands.get('ros_shell', []):
        if word.startswith('params_file:='):
            path = Path(word.split(':=',1)[1])
            yaml_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = dict(profile=profile, policy=policy, commands=commands,
                    inputs=inputs, ros_parameter_files=yaml_hashes)
    digest = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    payload = dict(schema='eval-storage-contract/v1', sha256=digest, contract=contract)
    base.mkdir(parents=True,exist_ok=True)
    temp = base/('.contract-'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(payload,indent=2))
    try:
        try:
            os.link(temp,base/'.eval_contract.json')
        except FileExistsError:
            pass
    finally:
        temp.unlink()
    original = json.loads((base/'.eval_contract.json').read_text())
    group = base if original['sha256'] == digest else base/'variants'/('config_'+digest[:16])
    group.mkdir(parents=True,exist_ok=True)
    if group != base:
        (group/'contract.json').write_text(json.dumps(payload,indent=2))
    destination = group/'episodes'
    destination.mkdir(exist_ok=True)
    manifest['recording']['ros_episode_dir'] = str(destination)
    manifest['recording']['storage_contract_sha256'] = digest
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest,indent=2))
    temporary.replace(manifest_path)
    return destination

def publish(run, profile):
    run = Path(run).resolve()
    manifest = json.loads((run / 'launch_manifest.json').read_text())
    hub = hub_root()
    hub.mkdir(parents=True, exist_ok=True)
    data = {'run_dir': str(run), 'profile': profile, 'published_ns': time.time_ns(),
            'policy': manifest['policy']}
    temporary = hub / ('.active_eval.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(hub / '.active_eval.json')

def read_active(hub):
    try:
        data = json.loads((Path(hub) / '.active_eval.json').read_text())
        run = Path(data['run_dir'])
        if not run.is_absolute() or not (run / 'launch_manifest.json').is_file():
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None

def run_episode(run):
    candidates = []
    for file, meta in episode_files(run):
        try:
            if meta.get('accepted_start'):
                candidates.append((meta['start_ns'], file.parent, meta))
        except (OSError, ValueError, KeyError):
            continue
    return max(candidates, key=lambda item: item[0], default=None)

if __name__ == '__main__':
    if sys.argv[1] == '--prepare':
        destination = prepare_shared(sys.argv[2], sys.argv[3])
        publish(sys.argv[2], sys.argv[3])
        print(destination)
    else:
        publish(sys.argv[1], sys.argv[2])
