class TestException(Exception):
    def __init__(self, message, x, y):
        super().__init__(message + str(x) + str(y))


def func():
    raise TestException('message', x=0, y=2)


try:
    func()
except TestException as te:
    print(te)
