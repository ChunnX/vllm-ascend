# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run without vLLM/NPU/pytest: python -m unittest <module> -v."""

import unittest

import numpy as np

from tests.ut.helpers.dcut_reference import recurrent_reference, speculative_conv_reference


class TestDcutCPUReference(unittest.TestCase):
    def test_conv_hand_computed_rollback_and_tail(self):
        state = np.arange(1, 13, dtype=float).reshape(2, 6, 1)
        before = state.copy()
        output, updated = speculative_conv_reference(
            np.array([[6.0], [7.0], [99.0]]),
            np.arange(1, 5).reshape(4, 1),
            state,
            np.array([0, 2, 2]),
            np.array([[0, 1, 1], [-1, -1, -1]]),
            np.array([3, 0]),  # Previous accepted > current query width.
            silu=False,
        )
        np.testing.assert_array_equal(output[:, 0], [50, 60, 0])
        np.testing.assert_array_equal(updated[0, :, 0], [4, 5, 6, 7, 5, 6])
        np.testing.assert_array_equal(updated[1], before[1])
        np.testing.assert_array_equal(state, before)

    def test_conv_silu_and_bias(self):
        output, _ = speculative_conv_reference(
            np.array([[2.0]]),
            np.ones((4, 1)),
            np.zeros((1, 4, 1)),
            np.array([0, 1]),
            np.array([0]),
            np.array([1]),
            bias=np.array([-3.0]),
        )
        self.assertAlmostEqual(output[0, 0], -1.0 / (1.0 + np.e))

    def test_conv_cross_round_rejects_suffix_and_reorders(self):
        weight = np.ones((4, 1))
        initial = np.zeros((2, 10, 1))
        _, first = speculative_conv_reference(
            np.array([[1.0], [2.0], [100.0], [4.0], [200.0]]),
            weight,
            initial,
            np.array([0, 3, 5]),
            np.array([0, 1]),
            np.array([1, 1]),
            silu=False,
        )
        out, second = speculative_conv_reference(
            np.array([[5.0], [3.0]]),
            weight,
            first,
            np.array([0, 1, 2]),
            np.array([1, 0]),
            np.array([1, 2]),
            silu=False,
        )
        # Request 1 commits 4; request 0 commits 1,2. Reject 200 and 100.
        np.testing.assert_array_equal(out[:, 0], [9, 6])
        np.testing.assert_array_equal(second[0, :3, 0], [1, 2, 3])
        np.testing.assert_array_equal(second[1, :3, 0], [0, 4, 5])

    def test_recurrent_hand_computed_accepted_not_clamped(self):
        state = np.array([9.0, 8.0, 2.0, 7.0]).reshape(4, 1, 1, 1)
        before = state.copy()
        out, updated = recurrent_reference(
            np.ones((3, 1, 1)),
            np.full((3, 1, 1), 0.5),
            np.array([3.0, 1.0, 99.0]).reshape(3, 1, 1),
            state,
            np.array([[0.5], [0.25], [1.0]]),
            np.array([[np.log(0.5)], [0.0], [0.0]]),
            np.array([0, 2, 2]),
            np.array([[0, 1, 2, 3], [-1, -1, -1, -1]]),
            np.array([3, 0]),
            scale=2.0,
        )
        np.testing.assert_allclose(out[:, 0, 0], [3.25, 3.296875, 0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(updated[:, 0, 0, 0], [1.625, 1.6484375, 2, 7], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(state, before)

    def test_recurrent_grouped_heads(self):
        # beta=0, g=0 means state is unchanged; isolate q/head mapping.
        out, _ = recurrent_reference(
            np.array([[[2.0], [3.0]]]),
            np.zeros((1, 2, 1)),
            np.zeros((1, 4, 1)),
            np.arange(1, 5, dtype=float).reshape(1, 4, 1, 1),
            np.zeros((1, 4)),
            np.zeros((1, 4)),
            np.array([0, 1]),
            np.array([[0]]),
            np.array([1]),
            scale=1.0,
        )
        np.testing.assert_array_equal(out[0, :, 0], [2, 4, 9, 12])

    def test_recurrent_cross_round_committed_prefix(self):
        rng = np.random.default_rng(17)
        q, k = (rng.normal(size=(4, 1, 2)) for _ in range(2))
        v = rng.normal(size=(4, 1, 3))
        state = rng.normal(size=(8, 1, 3, 2))
        beta, g = np.full((4, 1), 0.3), np.full((4, 1), -0.2)
        table = np.array([[5, 1, 7, 3]])  # Deliberately non-contiguous physical rows.
        _, first = recurrent_reference(
            q[:3],
            k[:3],
            v[:3],
            state,
            beta[:3],
            g[:3],
            np.array([0, 3]),
            table,
            np.array([1]),
            scale=1.0,
        )
        out, second = recurrent_reference(
            q[3:],
            k[3:],
            v[3:],
            first,
            beta[3:],
            g[3:],
            np.array([0, 1]),
            table,
            np.array([2]),
            scale=1.0,
        )
        # Independent algebraic form, recompute only committed t0,t1 plus t3.
        h = state[5, 0].copy()
        for t in (0, 1, 3):
            kt = k[t, 0]
            h = np.exp(g[t, 0]) * h @ (np.eye(2) - beta[t, 0] * np.outer(kt, kt)) + beta[t, 0] * np.outer(v[t, 0], kt)
        np.testing.assert_allclose(second[5, 0], h, atol=1e-14)
        np.testing.assert_allclose(out[0, 0], h @ q[3, 0], atol=1e-14)
        np.testing.assert_array_equal(second[[0, 2, 4, 6]], state[[0, 2, 4, 6]])

    def test_reject_undersized_speculative_conv_state(self):
        # 8 query tokens require 2 + 8 history slots for width-4 speculative conv.
        with self.assertRaisesRegex(ValueError, "conv state"):
            speculative_conv_reference(
                np.ones((8, 1)),
                np.ones((4, 1)),
                np.zeros((1, 8, 1)),
                np.array([0, 8]),
                np.array([0]),
                np.array([1]),
            )

    def test_all_inactive_rows_do_not_read_sentinel_or_accepted(self):
        out, state = recurrent_reference(
            np.ones((2, 1, 1)),
            np.ones((2, 1, 1)),
            np.ones((2, 1, 1)),
            np.ones((2, 1, 1, 1)),
            np.ones((2, 1)),
            np.zeros((2, 1)),
            np.array([0, 0, 0]),
            np.full((2, 2), -1),
            np.zeros(2, dtype=int),
            scale=1.0,
        )
        np.testing.assert_array_equal(out, 0)
        np.testing.assert_array_equal(state, 1)


if __name__ == "__main__":
    unittest.main()
