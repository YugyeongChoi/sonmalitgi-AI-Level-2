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
