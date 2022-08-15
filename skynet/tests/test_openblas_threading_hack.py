import pytest


def test_openblas_thread_hack():
    # We already have numpy imported in our tests.
    # Unfortunately unloading a module isn't really supported in python
    # https://bugs.python.org/issue9072
    # So all we check is that we correctly raises an error if numpy
    # is already imported.
    with pytest.raises(RuntimeError):
        # pylint: disable=unused-variable,unused-import,import-outside-toplevel
        import skynet.utils.openblas_thread_hack  # noqa


# pylint: enable=unused-variable,unused-import,import-outside-toplevel
