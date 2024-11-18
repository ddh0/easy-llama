# test_model.py
# https://github.com/ddh0/easy-llama/

"""Unit tests for the Model class"""

import unittest
from ..model import Model, ExceededContextLengthException

class TestModel(unittest.TestCase):

    def setUp(self):
        self.model_path = 'path/to/model.gguf'
        self.model = Model(self.model_path, context_length=512, n_gpu_layers=0)

    def test_load_model(self):
        self.assertTrue(self.model.is_loaded())
        self.assertEqual(self.model.context_length, 512)

    def test_tokenize(self):
        tokens = self.model.tokenize("Hello, world!")
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, int) for token in tokens))

    def test_detokenize(self):
        tokens = self.model.tokenize("Hello, world!")
        text = self.model.detokenize(tokens)
        self.assertEqual(text, "Hello, world!")

    def test_generate(self):
        response = self.model.generate("Hello, world!")
        self.assertIsInstance(response, str)
        self.assertTrue(response)

    def test_generate_exceeded_context(self):
        long_prompt = "a" * (self.model.context_length + 1)
        with self.assertRaises(ExceededContextLengthException):
            self.model.generate(long_prompt)

if __name__ == '__main__':
    unittest.main()