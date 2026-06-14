from typing import Protocol, Sized


class Sliceable(Sized, Protocol):
    def __getitem__[T: Sliceable](self: T, key: slice, /) -> T: ...
