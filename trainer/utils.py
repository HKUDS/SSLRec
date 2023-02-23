class DisabledSummaryWriter:
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self
