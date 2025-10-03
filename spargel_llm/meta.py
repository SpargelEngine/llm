from functools import wraps


def ai_marker(human_checked: bool = False, *, tag=None):
    """This decorator marks a function that has AI-generated content."""

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not human_checked:
                raise Exception(
                    f"{tag if tag is not None else func}: AI-generated content has not been marked as examined by human yet."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorate


def ai_marker_class(human_checked: bool = False):
    """This decorator marks a class that has AI-generated content."""

    def decorate(cls):
        cls.__init__ = ai_marker(human_checked, tag=cls)(cls.__init__)
        return cls

    return decorate
