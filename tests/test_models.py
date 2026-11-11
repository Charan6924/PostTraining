"""Loader contract tests with heavyweight/download dependencies isolated."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


def load_module():
    torch = types.ModuleType('torch')
    torch.bfloat16, torch.float16, torch.float32 = object(), object(), object()
    transformers = types.ModuleType('transformers')
    for name in ('AutoModelForCausalLM', 'AutoTokenizer', 'set_seed'):
        setattr(transformers, name, Mock())
    transformers.PreTrainedModel = object
    transformers.PreTrainedTokenizerBase = object
    spec = importlib.util.spec_from_file_location('models_under_test', Path(__file__).parents[1] / 'src/posttraining/models.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}):
        spec.loader.exec_module(module)
    return module


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.config = dict(model_name_or_path='Qwen/Qwen3-1.7B', revision='main', seed=42,
                           torch_dtype='bfloat16', enable_thinking=False, finetuning='full',
                           gradient_checkpointing=True)

    def test_every_missing_key_fails_before_loading(self):
        for key in self.config:
            with self.subTest(key=key):
                config = self.config.copy()
                del config[key]
                with self.assertRaisesRegex(ValueError, key):
                    self.module.load_model_and_tokenizer(config, training=True)
        self.module.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.module.AutoTokenizer.from_pretrained.assert_not_called()

    def test_invalid_values_fail_before_loading(self):
        for key, value in [('seed', True), ('revision', ''), ('torch_dtype', 'auto'),
                           ('finetuning', 'lora'), ('enable_thinking', 'false'),
                           ('gradient_checkpointing', None), ('model_name_or_path', 12)]:
            with self.subTest(key=key):
                config = self.config | {key: value}
                with self.assertRaises((ValueError, TypeError)):
                    self.module.load_model_and_tokenizer(config, training=True)
        self.module.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_training_returns_objects_and_enables_checkpointing(self):
        model, tokenizer = self.module.load_model_and_tokenizer(self.config, training=True)
        self.assertIs(model, self.module.AutoModelForCausalLM.from_pretrained.return_value)
        self.assertIs(tokenizer, self.module.AutoTokenizer.from_pretrained.return_value)
        self.module.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            self.config['model_name_or_path'], revision='main', torch_dtype=self.module.torch.bfloat16)
        self.module.AutoTokenizer.from_pretrained.assert_called_once_with(self.config['model_name_or_path'], revision='main')
        model.train.assert_called_once_with()
        model.gradient_checkpointing_enable.assert_called_once_with()
        self.assertFalse(model.config.use_cache)

    def test_inference_does_not_enable_checkpointing(self):
        model, _ = self.module.load_model_and_tokenizer(self.config, training=False)
        model.eval.assert_called_once_with()
        model.gradient_checkpointing_enable.assert_not_called()
        self.assertTrue(model.config.use_cache)

    def test_training_flag_must_be_boolean(self):
        with self.assertRaises(TypeError):
            self.module.load_model_and_tokenizer(self.config, training='false')


if __name__ == '__main__':
    unittest.main()
