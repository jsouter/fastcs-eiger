from typing import Callable, TypeVar

T = TypeVar("T")


def partition(
    elements: list[T], predicate: Callable[[T], bool]
) -> tuple[list[T], list[T]]:
    """Split a list of elements in two based on predicate.

    If the predicate returns ``True``, the element will be placed in the truthy list,
    if it does not, it will be placed in the falsy list.

    Args:
        elements: List of T
        predicate: Predicate to filter the list with

    Returns:
        (truthy, falsy)

    """
    truthy: list[T] = []
    falsy: list[T] = []
    for parameter in elements:
        if predicate(parameter):
            truthy.append(parameter)
        else:
            falsy.append(parameter)

    return truthy, falsy
