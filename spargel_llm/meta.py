from functools import wraps


def ai_marker(human_checked: bool = False):
    def decorate(obj):
        if isinstance(obj, type):
            orig_init = obj.__init__

            @wraps(orig_init)
            def wrapped_init(self, *args, **kwargs):
                if not human_checked:
                    raise Exception(
                        f"{obj}: Class contains AI-generated content, which has not been marked as examined by human yet."
                    )
                return orig_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init
            return obj

        else:

            @wraps(obj)
            def wrapper(*args, **kwargs):
                if not human_checked:
                    raise Exception(
                        f"{obj}: Object contains AI-generated content, which has not been marked as examined by human yet."
                    )
                return obj(*args, **kwargs)

            return wrapper

    return decorate
