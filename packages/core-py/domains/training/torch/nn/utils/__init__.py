class clip_grad_norm_:
    @staticmethod
    def __call__(parameters, max_norm, norm_type=2):
        return 0.0


class clip_grad_value_:
    @staticmethod
    def __call__(parameters, clip_value):
        pass
