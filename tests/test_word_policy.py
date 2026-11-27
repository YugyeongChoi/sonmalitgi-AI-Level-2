import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deployment'))
from ai_server.evaluation import Evaluator
from ai_server.model import Prediction


def test_word_practice_policy_without_claiming_calibration():
    judge = Evaluator('WORD1290', threshold=.5, stable=1, require_calibrated=False)
    assert judge.update(Prediction('WORD1290', .499, True, False))['result'] == 'RETRY'
    assert judge.update(Prediction('WORD1290', .5, True, False))['result'] == 'PASS'
    assert judge.update(Prediction('WORD1191', .8, True, False))['result'] == 'FAIL'
    assert judge.update(Prediction(None, 0., False, False))['result'] == 'CAPTURE_ERROR'


def test_other_models_still_require_calibration():
    judge = Evaluator('WORD1290')
    for _ in range(4):
        assert judge.update(Prediction('WORD1290', .99, True, False))['result'] == 'RETRY'


def test_sentence_sequence_requires_three_consecutive_predictions():
    from collections import deque
    from ai_server.sequence import WordSequence
    sequence = WordSequence('SEN0009')
    frames = deque(steering_frames())
    driving = Prediction('운전', .99, True, False)
    for _ in range(2):
        assert sequence.update(driving, frames)['result'] == 'RETRY'
        assert sequence.stage == 0
    sequence.update(driving, frames)
    assert sequence.stage == 1
    slow = Prediction('천천히', .99, True, False)
    assert sequence.update(slow, frames)['result'] == 'RETRY'
    sequence.update(Prediction('빨리', .99, True, False), frames)
    for _ in range(2):
        assert sequence.update(slow, frames)['result'] == 'RETRY'
    assert sequence.update(slow, frames)['result'] == 'PASS'


def test_consecutive_progress_tracks_target_and_resets():
    judge = Evaluator('목표', threshold=.9, stable=3, require_calibrated=False)
    good = Prediction('목표', .95, True, False)
    assert judge.update(good)['stable_count'] == 1
    assert judge.update(good)['stable_count'] == 2
    assert judge.update(Prediction('다른말', .99, True, False))['stable_count'] == 0
    assert judge.update(good)['stable_count'] == 1
    assert judge.update(Prediction('목표', .8, True, False))['stable_count'] == 0
    for expected in (1, 2, 3):
        result = judge.update(good)
        assert result['stable_count'] == expected
        assert result['stable_required'] == 3
        assert result['stable_target'] == '목표'
    assert result['result'] == 'PASS'

def fist_frame(open_hand=False):
    p=[dict(x=.5,y=.7,z=0) for _ in range(21)]
    p[0]=dict(x=.5,y=.8,z=0)
    for i,x in ((5,.43),(9,.48),(13,.53),(17,.58)):
        p[i]=dict(x=x,y=.62,z=0);p[i+1]=dict(x=x,y=.53,z=0)
        p[i+2]=dict(x=x,y=.46 if open_hand else .58,z=0 if open_hand else -.06)
        p[i+3]=dict(x=x,y=.40 if open_hand else .65,z=0 if open_hand else -.04)
    return dict(left_hand=p,right_hand=p,image_width=640,image_height=480)


def test_open_hand_high_score_never_advances_driving():
    from collections import deque
    from ai_server.sequence import WordSequence
    sequence = WordSequence('SEN0009')
    frames = deque([fist_frame(True)]*8)
    for _ in range(10):
        result = sequence.update(Prediction('운전', .999, True, False),frames)
        assert result['stable_count'] == 0
        assert result['recognized_words'] == []
        assert sequence.stage == 0
    frames = deque(steering_frames())
    assert sequence.update(Prediction('운전', .99, True, False),frames)['stable_count'] == 1
    frames.append(fist_frame(True))
    assert sequence.update(Prediction('운전', .99, True, False),frames)['stable_count'] == 0
    assert sequence.stage == 0


def test_unknown_or_missing_hand_cannot_confirm_driving():
    from collections import deque
    from ai_server.sequence import WordSequence
    sequence = WordSequence('SEN0011')
    frame = fist_frame();frame['right_hand'] = []
    result = sequence.update(Prediction('운전', .99, True, False),deque([frame]*8))
    assert result['stable_count'] == 0
    assert sequence.stage == 0

def steering_frames(mode='steer'):
    import math
    frames=[]
    for i in range(8):
        frame=fist_frame()
        t=math.radians(25*math.sin(i*math.pi/3)) if mode=='steer' else 0
        for key,sign in [('left_hand',-1),('right_hand',1)]:
            dx=sign*.18*math.cos(t)
            dy=sign*.18*640/480*math.sin(t)
            if mode=='translate': dx+=i*.01
            if mode=='one_hand' and sign==1: dx+=i*.02
            frame[key]=[dict(p,x=p['x']+dx,y=p['y']+dy) for p in frame[key]]
        frame['timestamp_ms']=i*125
        frames.append(frame)
    return frames


def test_motion_gate_rejects_static_fists_and_common_translation():
    from collections import deque
    from ai_server.sequence import WordSequence
    from ai_server.fist_gate import driving_motion_visible
    assert driving_motion_visible(steering_frames())
    for mode in ['static','translate','one_hand']:
        frames=deque(steering_frames(mode))
        assert not driving_motion_visible(frames)
        sequence=WordSequence('SEN0009')
        for _ in range(8):
            result=sequence.update(Prediction('운전',.999,True,False),frames)
            assert sequence.stage==0 and result['stable_count']==0


def test_motion_gate_rejects_stale_gaps_and_resets_consecutive_count():
    from collections import deque
    from ai_server.fist_gate import driving_motion_visible
    from ai_server.sequence import WordSequence
    frames=steering_frames();frames[-1]['timestamp_ms']+=1000
    assert not driving_motion_visible(frames)
    sequence=WordSequence('SEN0011')
    assert sequence.update(Prediction('운전',.99,True,False),deque(steering_frames()))['stable_count']==1
    assert sequence.update(Prediction('운전',.99,True,False),deque(steering_frames('static')))['stable_count']==0
    assert sequence.stage==0


def test_middle_counts_nonconsecutive_successes_without_reset():
    from ai_server.evaluation import CumulativeEvaluator
    judge=CumulativeEvaluator('WORD1290')
    good=Prediction('WORD1290',.8,True,False)
    assert judge.update(good)['success_count']==1
    for p in [Prediction('WORD1191',.99,True,False),Prediction(None,0,False,False),Prediction('WORD1290',.49,True,False)]:
        r=judge.update(p)
        assert r['success_count']==1 and r['result']=='RETRY'
    assert judge.update(good)['success_count']==1
    rearm(judge,0)
    assert judge.update(good)['success_count']==2
    rearm(judge,2000)
    assert judge.update(good)['result']=='PASS'
    assert judge.update(good)['success_count']==3
    assert CumulativeEvaluator('WORD1290').success_count==0


def rest_frame(timestamp):
    p=[dict(x=.5,y=.3,z=0,visibility=1) for _ in range(33)]
    p[11]['x']=.65;p[12]['x']=.35
    p[15]['y']=.95;p[16]['y']=.95
    return dict(pose=p,left_hand=[],right_hand=[],image_width=640,image_height=480,timestamp_ms=timestamp)

def rearm(judge,start):
    for i in range(6):judge.observe_frame(rest_frame(start+i*125))
    assert judge.armed

def test_middle_hold_or_detection_loss_never_rearms():
    from ai_server.evaluation import CumulativeEvaluator
    judge=CumulativeEvaluator('WORD1290');good=Prediction('WORD1290',.99,True,False)
    judge.update(good)
    for i in range(20):
        judge.observe_frame(dict(timestamp_ms=i*125,pose=[],left_hand=[],right_hand=[]))
        assert judge.update(good)['success_count']==1
    assert not judge.armed
    assert not judge.observe_frame(rest_frame(3000))
    assert not judge.observe_frame(rest_frame(4000))
    assert not judge.armed
    rearm(judge,5000)
    assert judge.update(good)['success_count']==2

def test_manual_prepare_does_not_add_success_or_reset_total():
    from ai_server.evaluation import CumulativeEvaluator
    judge=CumulativeEvaluator('WORD1191');good=Prediction('WORD1191',.99,True,False)
    assert not judge.prepare_next()
    assert judge.update(good)['success_count']==1
    assert judge.prepare_next()
    assert judge.success_count==1 and judge.armed
    assert not judge.prepare_next()
    assert judge.update(good)['success_count']==2
    assert judge.prepare_next()
    assert judge.update(good)['result']=='PASS'
    assert not judge.prepare_next()

def test_detected_lowered_hands_can_rearm_when_pose_wrists_are_occluded():
    from ai_server.evaluation import CumulativeEvaluator
    judge=CumulativeEvaluator('WORD1191');judge.update(Prediction('WORD1191',.9,True,False))
    for i in range(6):
        frame=rest_frame(i*125)
        frame['pose'][15]['visibility']=0;frame['pose'][16]['visibility']=0
        frame['left_hand']=[dict(x=.3,y=.8,z=0)]*21
        frame['right_hand']=[dict(x=.7,y=.8,z=0)]*21
        judge.observe_frame(frame)
    assert judge.armed and judge.success_count==1
