"""
수어 구간 처리 — "추후 결정"이라 교체 가능한 형태로 둔다 (KSL_SEGMENTER).

학습 데이터는 수어 동작 구간만 잘린 클립(앞뒤 3프레임 여유)이므로, 앱 입력도 비슷하게 잘려야
정확도가 유지된다. 어떤 방식을 쓰든 `__call__(arr) -> (start, end)` 만 지키면 된다.

  none  : 클라이언트가 보낸 start~end 그대로 (버튼 / 카운트다운 방식)
  hands : 앞뒤의 "쉬는 손만 있는 프레임"을 잘라냄 (실험용, 학습 때의 rest_y 기준 재사용)
  (예정) 자동 감지: WebSocket 스트림에서 시작·끝을 서버가 판단 → session에서 프레임마다 호출
"""
import numpy as np

from word_dataset import B_LWRI, B_RWRI, Config, interp_short_gaps, mask_low_conf, normalize


class FullRange:
    def __call__(self, arr):
        return 0, len(arr)


class TrimRestingHands:
    """한 손이라도 손목이 rest_y보다 위(수어 공간)에 있는 첫·마지막 프레임 ± margin."""

    def __init__(self, rest_y=None, margin=3, conf_thr=0.3):
        cfg = Config()
        self.rest_y = cfg.rest_y if rest_y is None else rest_y
        self.margin, self.conf_thr = margin, conf_thr

    def __call__(self, arr):
        xy = normalize(interp_short_gaps(mask_low_conf(arr, self.conf_thr), 5))
        y = np.stack([xy[:, B_LWRI, 1], xy[:, B_RWRI, 1]], 1)
        active = np.where((np.nan_to_num(y, nan=np.inf) < self.rest_y).any(1))[0]
        if len(active) == 0:
            return 0, len(arr)
        return max(0, active[0] - self.margin), min(len(arr), active[-1] + 1 + self.margin)


def make_segmenter(name: str):
    return {"none": FullRange, "hands": TrimRestingHands}[name]()
