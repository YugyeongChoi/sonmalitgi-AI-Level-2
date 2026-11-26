"""운전 단계의 열린 손 오통과를 막는 서버 수형 확인. 동작 전체 검증은 아님."""
import math

def hand_is_fist(points, width, height):
    if len(points) != 21 or not width or not height:
        return False
    try:
        p = [(q['x']*width, q['y']*height, q['z']*width) for q in points]
        if not all(math.isfinite(v) for q in p for v in q): return False
        def sub(a,b): return tuple(x-y for x,y in zip(a,b))
        def norm(a): return math.sqrt(sum(x*x for x in a))
        palm = norm(sub(p[9],p[0]))
        if palm < max(width,height)*.035: return False
        curled = 0
        for a,b,c,d in ((5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)):
            lengths = [norm(sub(p[i],p[j]))/palm for i,j in ((a,b),(b,c),(c,d))]
            if any(v < .06 or v > 1.1 for v in lengths): return False
            u,v = sub(p[a],p[b]),sub(p[c],p[b])
            denominator = norm(u)*norm(v)
            if denominator <= 1e-7: return False
            angle = math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(u,v))/denominator))))
            reach = norm(sub(p[d],p[0]))/norm(sub(p[b],p[0]))
            if angle <= 110 and reach <= 1.1: curled += 1
        return curled >= 3
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False

def driving_fists_visible(frames):
    recent = list(frames)[-3:]
    if len(recent) < 3: return False
    for frame in recent:
        if not isinstance(frame,dict): return False
        if not all(hand_is_fist(frame.get(key,[]), frame.get('image_width'), frame.get('image_height'))
                   for key in ('left_hand','right_hand')): return False
    return True


def driving_motion_visible(frames):
    """양손 손목 연결선의 왕복 회전과 두 손 이동을 확인한다. 정지·평행 이동은 제외."""
    samples = list(frames)[-8:]
    if len(samples) < 8 or not driving_fists_visible(samples): return False
    try:
        times = [f['timestamp_ms'] for f in samples]
        gaps = [b-a for a,b in zip(times,times[1:])]
        if any(not math.isfinite(t) for t in times) or any(g<=0 or g>500 for g in gaps): return False
        if not 500 <= times[-1]-times[0] <= 2500: return False
        left,right,scales,angles,distances = [],[],[],[],[]
        fist_frames = 0
        for f in samples:
            w,h = f['image_width'],f['image_height']
            if any(len(f.get(k,[])) != 21 for k in ('left_hand','right_hand')): return False
            def xy(p): return (p['x']*w,p['y']*h)
            def distance(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
            l,r = xy(f['left_hand'][0]),xy(f['right_hand'][0])
            if not all(math.isfinite(v) for p in (l,r) for v in p): return False
            left.append(l);right.append(r)
            scales.append((distance(l,xy(f['left_hand'][9]))+distance(r,xy(f['right_hand'][9])))/2)
            distances.append(distance(l,r))
            angles.append(math.degrees(math.atan2(r[1]-l[1],r[0]-l[0])))
            fist_frames += all(hand_is_fist(f[k],w,h) for k in ('left_hand','right_hand'))
        if fist_frames < 6: return False
        scale = sorted(scales)[len(scales)//2]
        if scale<=0 or min(distances)<scale*.7 or max(distances)>min(distances)*1.8: return False
        deltas = [(b-a+180)%360-180 for a,b in zip(angles,angles[1:])]
        if any(abs(d)>60 for d in deltas): return False  # 좌우 오배정·급격한 추적 점프
        unwrapped=[0.0]
        for d in deltas: unwrapped.append(unwrapped[-1]+d)
        if max(unwrapped)-min(unwrapped)<15: return False
        # 한 방향으로 들기만 하는 동작 대신 양 방향 회전 증거가 모두 필요하다.
        if sum(d for d in deltas if d>2)<6 or -sum(d for d in deltas if d< -2)<6: return False
        for positions in (left,right):
            span=max(distance(a,b) for a in positions for b in positions)
            if span/scale < .35: return False
        centers=[((l[0]+r[0])/2,(l[1]+r[1])/2) for l,r in zip(left,right)]
        center_span=max(distance(a,b) for a in centers for b in centers)
        relative_span=max(distance((r[0]-l[0],r[1]-l[1]),(rr[0]-ll[0],rr[1]-ll[1]))
                          for l,r in zip(left,right) for ll,rr in zip(left,right))
        if center_span > relative_span*.65: return False
        return True
    except (KeyError,TypeError,ValueError,ZeroDivisionError):
        return False
