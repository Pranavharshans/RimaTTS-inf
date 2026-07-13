import unittest

import torch

from chatterbox.models.t3.preallocated_cache import PreallocatedDynamicLayer


class PreallocatedDynamicLayerTest(unittest.TestCase):
    def test_updates_in_place_and_returns_populated_prefix(self):
        layer = PreallocatedDynamicLayer(max_cache_len=5)
        first_key = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
        first_value = first_key + 100
        keys, values = layer.update(first_key, first_value)
        key_data_ptr = layer.keys.data_ptr()
        value_data_ptr = layer.values.data_ptr()

        second_key = torch.full((1, 2, 1, 3), 42.0)
        second_value = torch.full((1, 2, 1, 3), 84.0)
        keys, values = layer.update(second_key, second_value)

        self.assertEqual(layer.get_seq_length(), 3)
        self.assertEqual(tuple(keys.shape), (1, 2, 3, 3))
        self.assertEqual(tuple(values.shape), (1, 2, 3, 3))
        self.assertEqual(layer.keys.data_ptr(), key_data_ptr)
        self.assertEqual(layer.values.data_ptr(), value_data_ptr)
        torch.testing.assert_close(keys[:, :, :2], first_key)
        torch.testing.assert_close(values[:, :, :2], first_value)
        torch.testing.assert_close(keys[:, :, 2:], second_key)
        torch.testing.assert_close(values[:, :, 2:], second_value)

    def test_capacity_is_enforced(self):
        layer = PreallocatedDynamicLayer(max_cache_len=1)
        state = torch.zeros((1, 1, 1, 2))
        layer.update(state, state)
        with self.assertRaisesRegex(ValueError, "capacity"):
            layer.update(state, state)


if __name__ == "__main__":
    unittest.main()
