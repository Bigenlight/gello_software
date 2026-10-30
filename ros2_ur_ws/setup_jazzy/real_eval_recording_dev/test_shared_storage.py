import os, sys, json, time, tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.extend(['/home/junhyeong/gello_software_jazzy/ros2_ur_ws', '/home/junhyeong/gello_software_jazzy/ros2_ur_ws/setup_jazzy/real_eval_recording_dev'])
from recording import EpisodeWriter, tag_episode
from eval_log_hub import prepare_shared, episode_root, run_episode, publish
from placement_metadata import PlacementMetadata
from pending_eval_tags import PendingTags
from camera_viewer_qt import Window
from PyQt5.QtWidgets import QApplication
from types import SimpleNamespace
import cv2, numpy as np

app=QApplication([])
with tempfile.TemporaryDirectory() as tmp:
    hub=Path(tmp); base=hub/'svf_cube_full_h8_step100000'
    os.environ['REAL_EVAL_LOG_HUB']=str(hub)
    def new_run(n,extra=''):
        run=base/'.sessions'/f'run_20260920T10450{n}Z_pid{n}'
        run.mkdir(parents=True)
        manifest={'policy':{'family':'svf','step':100000},
                  'commands':{'server_shell':f'python server.py --log-dir {run} --step 100000',
                              'ros_shell':'ros2 launch package launch.py '+extra},
                  'artifacts':{'inputs':{'checkpoint':{'path':'params_100000.pkl','sha256':'abc'}}},
                  'recording':{'ros_episode_dir':str(run/'episodes')}}
        (run/'launch_manifest.json').write_text(json.dumps(manifest))
        destination=prepare_shared(run,'svf_cube_full_h8_step100000')
        return run,destination
    run1,root1=new_run(1)
    run2,root2=new_run(2)
    assert root1==root2==base/'episodes'
    run3,root3=new_run(3,'rate:=20')
    assert root3!=root1 and root3.parent.parent.name=='variants'
    run4,root4=new_run(4,'rate:=20')
    assert root4==root3
    assert not (run2/'ep_0001.npz').exists()
    (run1/'ep_0001.npz').write_bytes(b'original server data')
    os.environ['REAL_EVAL_EPISODE_ROOT']=str(root1)
    os.environ['REAL_EVAL_RUN_DIR']=str(run1)
    request=time.monotonic_ns()
    writer=EpisodeWriter(run1/'episodes', {'accepted_start':True,'render':True,'method':'bc'})
    writer.start_wall_ns=1789893000123456000
    captured=PlacementMetadata(hub)
    captured.enqueue(run1,request,time.monotonic_ns(),{'split':'val','mode':'single_episode','takes':['sample']})
    tags=PendingTags(hub)
    tags.enqueue(writer.path,writer.id,'partial_success')
    ok,jpeg=cv2.imencode('.jpg',np.zeros((360,640,3),np.uint8)); assert ok
    stamp=time.monotonic_ns()
    writer.append_image('cam1',jpeg.tobytes(),stamp)
    writer.append_image('cam2',jpeg.tobytes(),stamp)
    time.sleep(.04)
    writer.stop('operator_hold'); assert writer.wait(5)
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        tags.process(); captured.process()
        selected=run_episode(run1)
        if selected and selected[1].name.endswith('__partial_success') and (selected[1]/'placement_reference.json').exists(): break
        time.sleep(.03)
    path=selected[1]
    assert path.parent==root1 and '_KST_'+writer.id in path.name and path.name.endswith('__partial_success')
    assert json.loads((path/'outcome.json').read_text())['success_score']==.5
    assert json.loads((path/'render_status.json').read_text())['status']=='complete'
    assert (path/'diagnostic.mp4').stat().st_size>0
    assert (path/'placement_reference.json').exists()
    assert run_episode(run2) is None  # same storage must not select previous CLI episode
    os.environ['REAL_EVAL_RUN_DIR']=str(run2)
    second=EpisodeWriter(run2/'episodes', {'accepted_start':True})
    second.stop(); second.wait(5)
    assert run_episode(run2)[2]['episode_id']==second.id
    assert len(list(root1.glob('ep_*')))==2
    assert (run1/'ep_0001.npz').read_bytes()==b'original server data'
    class Control:
        ready=True; result_text=None; result_expiry=0
        def __init__(self,label): self.label=label
        def in_flight(self): return False
    controls=[Control(x) for x in ['START','HOLD','RESET']]
    feed=SimpleNamespace(label='camera',frame=None,is_stalled=lambda:False,fps=lambda:30)
    publish(run2,'test')
    w=Window(SimpleNamespace(cam1=feed,cam2=feed),SimpleNamespace(_buttons=controls,update=lambda *a:None),str(hub))
    w.timer.stop(); w.refresh()
    assert w.selected[2]['episode_id']==second.id and 'Episode START:' in w.identity.text()
    publish(run3,'changed'); w.last_scan=0; w.refresh()
    assert w.selected is None
    w.overlay.shutdown(); w.close()
    # Old per-run layouts remain discoverable and relabelable.
    os.environ.pop('REAL_EVAL_EPISODE_ROOT'); os.environ.pop('REAL_EVAL_RUN_DIR')
    legacy=hub/'run_legacy'; (legacy/'episodes').mkdir(parents=True)
    (legacy/'launch_manifest.json').write_text(json.dumps({'recording':{}}))
    old=EpisodeWriter(legacy/'episodes',{'accepted_start':True}); old.stop(); old.wait(5)
    old_path=old.path.with_name('ep_'+old.id+'__unlabeled'); old.path.rename(old_path)
    assert run_episode(legacy)[2]['episode_id']==old.id
    assert tag_episode(old_path,'success').name=='ep_'+old.id+'__success'
print('PASS: shared storage across restarts, argument variants, private NPZ, START timestamp, render, partial tag, placement binding, GUI run isolation and legacy compatibility')
