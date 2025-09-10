Rank = int

MAX_RANK = 1_000_000
MAX_OFFSET = 1_000_000_000


def byte_pair_merge(ranks: dict[bytes, Rank], piece: bytes) -> list[int]:
    """
    Perform merges with the give ranks.

    Note: rank means priority for a merge, i.e. the lower the rank is, the more urgent the merge is.

    The input bytes are merged into segments.

    Args:
        ranks: a dictionary specifying ranks for some byte sequences
               Note: it is assume that `ranks[s] <= ranks[t]` if `s` is a prefix of `t`, and
                     every key in `ranks` can be written as the sum of another two keys in `ranks`.
        piece: the input byte sequence

    Return: the start position of each segment
    """
    # every item represents a part of `piece` with rank
    parts: list[tuple[int, Rank]] = []

    # index of the the merge point (in `parts`) with minimal rank, i.e. maximal merge priority
    min_rank: tuple[int, Rank] = (MAX_OFFSET, MAX_RANK)

    # iterate over the adjacent bytes
    for i in range(len(piece) - 1):
        # if the byte-pair does not exist in `ranks`, assign inf to rank
        rank = ranks.get(piece[i : i + 2], MAX_RANK)
        if rank < min_rank[1]:
            min_rank = (i, rank)
        parts.append((i, rank))
    parts.append((len(piece) - 1, MAX_RANK))
    # add a virtual merge point at the end
    parts.append((len(piece), MAX_RANK))

    # sanity check
    assert len(parts) == len(piece) + 1

    # get the rank of byte-pair formed by merge points `i`, `i+1`, `i+2`
    # note: this is called when `i` and `i+1` will be merged, or when `i+1` and `i+2` will be merged
    def get_rank(i: int) -> Rank:
        if i + 3 < len(parts):
            p = piece[parts[i][0] : parts[i + 3][0]]
            return ranks.get(p, MAX_RANK)
        else:
            return MAX_RANK

    # loop condition: there are byte-pairs that can be merged
    while not min_rank[1] == MAX_RANK:
        # the offset of the merge point with minimal rank
        i = min_rank[0]

        # we need to recompute the rank at the previous byte if there is one
        if i > 0:
            # only the rank is modified
            parts[i - 1] = (parts[i - 1][0], get_rank(i - 1))
        parts[i] = (parts[i][0], get_rank(i))
        # remove the next merge point, as it has been merged with `i`
        parts.pop(i + 1)

        # find the next merge point with minimal rank
        min_rank = (MAX_OFFSET, MAX_RANK)
        for i, (_, rank) in enumerate(parts[:-1]):
            if rank < min_rank[1]:
                min_rank = (i, rank)

    return [segment[0] for segment in parts[:-1]]
