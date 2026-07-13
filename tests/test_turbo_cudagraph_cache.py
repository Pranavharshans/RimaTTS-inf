import unittest
from types import SimpleNamespace

import torch

from chatterbox.models.t3.t3 import _pack_cudagraph_cache_outputs


class TurboCudagraphCacheTest(unittest.TestCase):
    def test_packs_cache_values_into_contiguous_shared_storage(self):
        layers = []
        expected = []
        for layer_index in range(3):
            keys = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
            keys = keys + layer_index * 100
            values = keys + 50
            layers.append(
                SimpleNamespace(
                    is_initialized=True,
                    keys=keys,
                    values=values,
                )
            )
            expected.append((keys.clone(), values.clone()))

        _pack_cudagraph_cache_outputs(SimpleNamespace(layers=layers))

        storage_ptr = layers[0].keys.untyped_storage().data_ptr()
        for layer, (expected_keys, expected_values) in zip(layers, expected):
            self.assertTrue(torch.equal(layer.keys, expected_keys))
            self.assertTrue(torch.equal(layer.values, expected_values))
            self.assertTrue(layer.keys.is_contiguous())
            self.assertTrue(layer.values.is_contiguous())
            self.assertEqual(layer.keys.untyped_storage().data_ptr(), storage_ptr)
            self.assertEqual(layer.values.untyped_storage().data_ptr(), storage_ptr)


if __name__ == "__main__":
    unittest.main()
