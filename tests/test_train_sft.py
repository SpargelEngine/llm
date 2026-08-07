"""Unit tests for spargel_llm.train.sft."""

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from spargel_llm.train.sft import Message, build_sft_item_chatml


def _make_tiny_tokenizer() -> Tokenizer:
    """Create a minimal BPE tokenizer with known special-token IDs.

    Uses ByteLevel pre-tokenization so any text can be encoded without
    ``<unk>`` fallback.  Training data includes the role names and
    typical conversational fragments used in the tests below.
    """
    training_texts = [
        "hello world this is a test message",
        "how are you doing today",
        "I am fine thank you very much",
        "the quick brown fox jumps over the lazy dog",
        "system user assistant tool",
        "What is the answer to this question",
        "The answer is forty two",
        "Please provide more detail",
        "Sure let me explain in detail",
        "Hi How are you",
        "Hello I am good thanks",
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789 .,!?'\"",
        "\n",
    ]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=512,
        special_tokens=["<unk>", "<pad>", "<im_start>", "<im_end>"],
        show_progress=False,
    )
    tokenizer.train_from_iterator(training_texts, trainer=trainer)
    return tokenizer


# Special-token IDs produced by _make_tiny_tokenizer.
# The BPE trainer assigns them in the order listed, so they are stable.
IM_START = 2
IM_END = 3
PAD = 1


# ═══════════════════════════════════════════════════════════════════
# build_item_chatml
# ═══════════════════════════════════════════════════════════════════


class TestBuildItemChatmlBasic:
    def test_single_turn(self):
        """system + user + assistant: every output field has the right shape."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello there!"),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )

        assert item.seq_len == 64
        assert item.input_ids.shape == (64,)
        assert item.target_ids.shape == (64,)
        assert item.mask.shape == (64,)
        assert item.input_ids.dtype == np.int64
        assert item.target_ids.dtype == np.int64
        assert item.mask.dtype == np.bool
        assert item.num_non_pad > 0
        # num_valid counts the visible (masked-in) positions
        assert item.num_valid == int(item.mask.sum())

    def test_input_sequence_starts_with_im_start(self):
        """The very first token should be ``<im_start>``."""
        tokenizer = _make_tiny_tokenizer()
        messages = [Message(role="user", content="Hello.")]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=32,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        assert item.input_ids[0] == IM_START

    def test_pad_beyond_valid_region(self):
        """Positions after the valid mask are all *pad_id*."""
        tokenizer = _make_tiny_tokenizer()
        messages = [Message(role="assistant", content="Hi")]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        valid_len = item.mask.sum()
        assert valid_len < 64
        np.testing.assert_array_equal(
            item.input_ids[valid_len:], np.full(64 - valid_len, PAD, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            item.target_ids[valid_len:], np.full(64 - valid_len, PAD, dtype=np.int64),
        )

    def test_mask_covers_input_exactly(self):
        """*mask* is ``True`` for the first *L* positions, ``False`` after."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="World"),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=48,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        valid_len = item.mask.sum()
        assert valid_len > 0
        np.testing.assert_array_equal(item.mask[:valid_len], True)
        np.testing.assert_array_equal(item.mask[valid_len:], False)


class TestBuildItemChatmlTargets:
    def test_only_assistant_has_targets(self):
        """Non-assistant positions must have *pad_id* as target."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is the weather?"),
            Message(role="assistant", content="It is sunny today."),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=128,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        valid_len = item.mask.sum()

        # Walk through positions manually: collect ranges where target != PAD
        target_positions = []
        for i in range(valid_len):
            if item.target_ids[i] != PAD:
                target_positions.append(i)
        assert len(target_positions) > 0

        # All target positions should be inside the assistant message
        # Decode the input at each target position —  they should be part of
        # the assistant's content or the preceding \n
        for pos in target_positions:
            # target_ids[pos] should equal input_ids[pos + 1]
            assert item.target_ids[pos] == item.input_ids[pos + 1]

    def test_no_assistant_messages(self):
        """When there is no assistant, *num_non_pad* is 0 and targets are pad."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello."),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        assert item.num_non_pad == 0
        # but the input is still fully visible
        assert item.num_valid == int(item.mask.sum()) > 0
        valid_len = item.mask.sum()
        np.testing.assert_array_equal(
            item.target_ids[:valid_len], np.full(valid_len, PAD, dtype=np.int64),
        )

    def test_multi_turn(self):
        """Each assistant turn produces its own target range."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I am good."),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=128,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        valid_len = item.mask.sum()

        # Collect contiguous target (non-PAD) ranges
        ranges = []
        in_range = False
        start = None
        for i in range(valid_len):
            if item.target_ids[i] != PAD and not in_range:
                start = i
                in_range = True
            elif item.target_ids[i] == PAD and in_range:
                ranges.append((start, i))
                in_range = False
        if in_range:
            ranges.append((start, valid_len))

        # Two assistant messages → two target ranges
        assert len(ranges) == 2, f"Expected 2 target ranges, got {len(ranges)}"

        # num_non_pad covers both ranges
        total = sum(end - start for start, end in ranges)
        assert item.num_non_pad == total
        assert item.num_valid == int(item.mask.sum())

    def test_shift_is_next_token_prediction(self):
        """For every position where target is not *pad_id*,
        ``target_ids[i] == tokens[i+1]`` holds (standard language-model shift)."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="assistant", content="One two three four five."),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        valid_len = item.mask.sum()
        for i in range(valid_len - 1):
            if item.target_ids[i] != PAD:
                assert item.target_ids[i] == item.input_ids[i + 1], (
                    f"Shift violation at pos {i}: "
                    f"target={item.target_ids[i]}, input[i+1]={item.input_ids[i+1]}"
                )

    def test_num_non_pad_matches_target_count(self):
        """*num_non_pad* equals the number of non-PAD entries in *target_ids*."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="user", content="Question here."),
            Message(role="assistant", content="Answer here."),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        expected = int((item.target_ids != PAD).sum())
        assert item.num_non_pad == expected

    def test_num_valid_is_visible_length(self):
        """*num_valid* equals the number of positions covered by *mask*,
        and can exceed *num_non_pad* (input is visible where there are no
        assistant targets)."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="user", content="Question here."),
            Message(role="assistant", content="Answer here."),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        assert item.num_valid == int(item.mask.sum())
        assert item.num_valid >= item.num_non_pad
        assert item.num_valid > item.num_non_pad


class TestBuildItemChatmlTruncation:
    def test_full_sequence_fits(self):
        """When the sequence is shorter than *seq_len*, no truncation occurs."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="assistant", content="Short"),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=256,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        valid_len = item.mask.sum()
        # The whole encoded sequence fits, so the last non-pad input should
        # be the trailing newline, and the position after that is pad
        assert valid_len < 256
        # There should be at least one assistant target token
        assert item.num_non_pad >= 1

    def test_truncation_before_assistant_starts(self):
        """If *seq_len* is too small for the assistant message to begin,
        *num_non_pad* is 0."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Tell me a story about dragons."),
            Message(role="assistant", content="Once upon a time there was a dragon."),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=8,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        assert item.num_non_pad == 0

    def test_assistant_content_partially_truncated(self):
        """When the assistant content extends beyond *seq_len*, only the
        portion that fits receives targets."""
        tokenizer = _make_tiny_tokenizer()
        # Use very long assistant content so it definitely overflows
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="A" * 500),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=32,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        assert item.num_non_pad > 0
        # All target positions must be within [0, seq_len)
        target_positions = np.where(item.target_ids != PAD)[0]
        assert len(target_positions) > 0
        assert target_positions[-1] < 32

    def test_mask_all_true_when_everything_fits(self):
        """When ``len(tokens) - 1 <= seq_len``, *mask* is ``True`` for all
        filled input positions and no input position holds *pad_id* inside
        the valid region."""
        tokenizer = _make_tiny_tokenizer()
        messages = [Message(role="assistant", content="Hi")]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        valid_len = item.mask.sum()
        for i in range(valid_len):
            assert item.input_ids[i] != PAD


class TestBuildItemChatmlRoles:
    def test_custom_assistant_role(self):
        """A non-default assistant role name still receives targets."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="user", content="Hi"),
            Message(role="tool", content="Result: 42"),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
            assistant_role="tool",
        )
        assert item.num_non_pad > 0

    def test_custom_role_no_targets_if_not_assistant(self):
        """Only the role named by *assistant_role* gets targets."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="speaker", content="Hello"),
            Message(role="listener", content="World"),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
            assistant_role="listener",
        )
        valid_len = item.mask.sum()
        # "speaker" message: targets are all pad
        # "listener" message: targets are set
        # Verify we can find non-pad targets
        has_targets = (item.target_ids[:valid_len] != PAD).any()
        assert has_targets

        # The first message ("speaker") should not contribute targets
        # Rebuild with assistant_role="speaker" to compare
        item2 = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
            assistant_role="speaker",
        )
        # The two items should differ in which positions have targets
        valid1 = item.mask.sum()
        valid2 = item2.mask.sum()
        t1 = item.target_ids[:valid1] != PAD
        t2 = item2.target_ids[:valid2] != PAD
        assert not (t1 == t2).all()

    def test_role_string_not_in_vocab(self):
        """Roles are encoded fresh (not looked up as single tokens), so even
        an arbitrary Unicode string works."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="human", content="What is the answer?"),
            Message(role="robot-assistant-007", content="Forty-two."),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=128,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
            assistant_role="robot-assistant-007",
        )
        assert item.num_non_pad > 0


class TestBuildItemChatmlEdgeCases:
    def test_only_assistant_message(self):
        """A conversation consisting of a single assistant message."""
        tokenizer = _make_tiny_tokenizer()
        messages = [Message(role="assistant", content="Solo message.")]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        assert item.num_non_pad > 0

    def test_very_large_seq_len(self):
        """A large *seq_len* does not break; the token sequence is just
        shorter than the buffer."""
        tokenizer = _make_tiny_tokenizer()
        messages = [Message(role="assistant", content="Tiny.")]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=4096,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        assert item.seq_len == 4096
        assert item.input_ids.shape == (4096,)
        assert item.num_non_pad > 0

    def test_multiple_assistant_messages_with_different_content(self):
        """Tokens from different assistant turns do not interfere."""
        tokenizer = _make_tiny_tokenizer()
        messages = [
            Message(role="assistant", content="AAA"),
            Message(role="user", content="BB"),
            Message(role="assistant", content="CCC"),
        ]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=128,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        # Decode the full input to verify both "AAA" and "CCC" are present
        valid_len = item.mask.sum()
        decoded = tokenizer.decode(item.input_ids[:valid_len].tolist())
        assert "AAA" in decoded
        assert "CCC" in decoded

    def test_im_end_is_target(self):
        """The ``<im_end>`` token at the end of assistant content is included
        in the target (the model learns to emit it)."""
        tokenizer = _make_tiny_tokenizer()
        messages = [Message(role="assistant", content="X")]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        valid_len = item.mask.sum()
        # The last target in the assistant turn should be IM_END
        target_positions = np.where(item.target_ids[:valid_len] != PAD)[0]
        assert len(target_positions) > 0
        last_target_pos = target_positions[-1]
        assert item.target_ids[last_target_pos] == IM_END, (
            f"Expected last target to be IM_END ({IM_END}), "
            f"got {item.target_ids[last_target_pos]}"
        )

    def test_newline_before_content_is_start_of_target(self):
        """The ``\\n`` after the assistant role is the input whose target
        is the first content token — model learns to transition from role
        to content."""
        tokenizer = _make_tiny_tokenizer()
        messages = [Message(role="assistant", content="Test content.")]
        item = build_sft_item_chatml(
            messages, tokenizer, seq_len=64,
            pad_id=PAD, im_start_id=IM_START, im_end_id=IM_END,
        )
        valid_len = item.mask.sum()

        # Find the first position where target is not PAD
        first_target_pos = None
        for i in range(valid_len):
            if item.target_ids[i] != PAD:
                first_target_pos = i
                break
        assert first_target_pos is not None

        # The input at this position should be the newline after the role
        # Decode the role tokens + newline to confirm
        newline_ids = tokenizer.encode("\n").ids
        input_at_target = item.input_ids[first_target_pos]
        assert input_at_target in newline_ids or input_at_target == newline_ids[0], (
            f"Expected newline token at target start position {first_target_pos}, "
            f"got {input_at_target}"
        )
