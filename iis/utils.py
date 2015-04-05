""" Utilitis
"""

def formatdoc(*args, **kwargs):
    def deco(method):
        """ Format a function or class method's documentation
        """
        try:
            method.__doc__ = method.__doc__.format(*args, **kwargs)
        except AttributeError:
            method.__func__.__doc__ = method.__func__.__doc__.format(*args, **kwargs)
        return method
    return deco
