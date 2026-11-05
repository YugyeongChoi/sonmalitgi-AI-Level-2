"""Use the complete recorded sign; body-relative rest trimming is unavailable."""
class FullRange:
    def __call__(self, arr):
        return 0, len(arr)


def make_segmenter(name: str):
    if name != "none":
        raise ValueError("Hands-only model supports KSL_SEGMENTER=none only")
    return FullRange()
