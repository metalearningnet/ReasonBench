import re
import os
import sys
import time
import json
import random
import openai
import logging
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from config import (
    CHOICE_MAP, DEFAULT_DATA_DIR, LLM_API_BASE, LLM_API_MODEL, COT_TOKEN_NAMES,
    COT_TOKENS, END_MARK, ENABLE_THINKING, SHOW_TRAJECTORY_ON_FAIL, SHOW_PROMPT, SHOW_RESPONSE, SHOW_ACTION,
    load_config, load_datasets_config, logger, dataset_names
)

for logger_name in ["openai", "httpx", "httpcore", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

def format_trajectory(steps: List[Dict[str, Any]], goal: str) -> str:
    lines = [f"Goal: {goal}\n"]
    for i, step in enumerate(steps):
        obs = step["observation"]
        if len(obs) > 200:
            obs = obs[:200] + "..."
        lines.append(f"Step {i+1}:")
        lines.append(f"  Obs: {obs}")
        lines.append(f"  Action: {step['action']}")
        lines.append(f"  Admissible: {step['admissible_commands'][:5]}..." if len(step['admissible_commands']) > 5 else f"  Admissible: {step['admissible_commands']}")
    return "\n".join(lines)

@dataclass
class GeneratorArguments:
    dataset: str = field(
        default="truthfulqa",
        metadata={
            "choices": dataset_names,
            "help": "Dataset to generate CoT data for"
        }
    )
    mode: str = field(
        default="train",
        metadata={
            "choices": ["train", "test"],
            "help": "Whether to generate train or test data"
        }
    )
    api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key for LLM service"}
    )
    api_base: Optional[str] = field(
        default=None,
        metadata={"help": "API endpoint URL"}
    )
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Model name. For OpenAI: 'gpt-4o', etc. For vLLM: HuggingFace model path"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for generated datasets"}
    )
    max_retries: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum retries for API calls"}
    )
    num_examples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of examples to generate"}
    )
    validate: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to validate CoT steps"}
    )
    dry_run: Optional[bool] = field(
        default=False,
        metadata={"help": "Test configuration without making API calls"}
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature for generation"}
    )
    backend: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["api", "vllm"],
            "help": "LLM inference backend: 'api' for unified remote API access, 'vllm' for local inference."
        }
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size for generation"}
    )
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to config file"}
    )
    debug: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable debug logging for troubleshooting"}
    )

class BaseLLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        generator_config = config.get('generator', {})
        llm_config = generator_config.get('api', {})
        max_prompt_length = config['train']['max_prompt_length']
        max_completion_length = config['train']['max_completion_length']
        
        self.max_tokens = max_prompt_length + max_completion_length
        self.frequency_penalty = generator_config.get('frequency_penalty', 0.0)
        self.presence_penalty = generator_config.get('presence_penalty', 0.0)
        self.stop_sequences = generator_config.get('stop_sequences', [])
        self.temperature = generator_config.get('temperature', 0.3)
        self.max_retries = generator_config.get('max_retries', 20)
        self.batch_size = generator_config.get('batch_size', 1)
        self.top_p = generator_config.get('top_p', 1.0)
        
        self.retry_max_wait = llm_config.get('retry_max_wait', 60)
        self.retry_min_wait = llm_config.get('retry_min_wait', 1)

        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0
    
    def get_response(self, prompt: str, max_tokens: int = None) -> str:
        raise NotImplementedError
    
    def get_responses(self, prompts: List[str], max_tokens: int = None) -> List[str]:
        raise NotImplementedError
    
    def get_cost_summary(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_output_tokens": self.total_output_tokens
        }
    
    def print_summary(self):
        summary = self.get_cost_summary()
        logger.info(f"Summary:")
        logger.info(f"  Total Requests: {summary['total_requests']}")
        logger.info(f"  Output Tokens: {summary['total_output_tokens']:,}")

class OpenAIClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        llm_config = config.get('generator', {}).get('api', {})
        
        self.api_key = llm_config.get('api_key', '')
        self.api_base = llm_config.get('api_base', LLM_API_BASE)
        self.model = llm_config.get('model', LLM_API_MODEL)
        self.timeout = llm_config.get('timeout', 60)
        
        self._configure_client()
        
        if not config.get('dry_run', False):
            self._test_connection()
    
    def _configure_client(self):
        if not self.api_key:
            logger.warning("No API key provided. Some LLM services may not require an API key.")
        
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": 0
        }
        
        if self.api_base != LLM_API_BASE:
            client_kwargs["base_url"] = self.api_base
        
        self.client = openai.OpenAI(**client_kwargs)
    
    def _test_connection(self):
        test_messages = [{"role": "user", "content": "Hello, are you working?"}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=test_messages,
                max_tokens=5
            )
            
            if response:
                logger.info(f"✓ Connected to OpenAI API at {self.api_base}")
                logger.info(f"✓ Using model: {self.model}")
                return True
        except Exception as e:
            logger.error(f"✗ Connection test failed: {e}")
            logger.info("Please check your configuration:")
            logger.info(f"  - API Base: {self.api_base}")
            logger.info(f"  - API Key: {'Set' if self.api_key else 'Not set'}")
            logger.info(f"  - Model: {self.model}")
            return False
    
    def _make_request(self, messages: List[Dict], max_tokens: int = None) -> Optional[Dict]:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
        
        if self.stop_sequences:
            params["stop"] = self.stop_sequences
        
        try:
            response = self.client.chat.completions.create(**params)
            self.total_requests += 1
            
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                output_tokens = usage.completion_tokens
                self.total_output_tokens += output_tokens
            
            return response
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_response(self, prompt: str, max_tokens: int = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(self.max_retries):
            try:
                response = self._make_request(messages, max_tokens)
                if response and response.choices:
                    return response.choices[0].message.content
                else:
                    logger.warning(f"Empty response, attempt {attempt + 1}/{self.max_retries}")
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                wait_time = min(self.retry_min_wait * (2 ** attempt), self.retry_max_wait)
                wait_time *= (0.5 + random.random())
                
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {e}")
                raise
        
        raise Exception(f"Failed to get response after {self.max_retries} attempts")
    
    def get_responses(self, prompts: List[str], max_tokens: int = None) -> List[str]:
        responses = []
        for prompt in tqdm(prompts, desc="Generating with OpenAI API"):
            try:
                response = self.get_response(prompt, max_tokens)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {e}")
                responses.append("")
        return responses

class VLLMClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        vllm_config = config.get('generator', {}).get('vllm', {})

        self.model = vllm_config.get('model')
        self.seed = config['common'].get('seed', 42)
        self.use_chat_template = vllm_config.get('use_chat_template', True)
        self.tensor_parallel_size = vllm_config.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = vllm_config.get('gpu_memory_utilization', 0.8)
        self.system_message = vllm_config.get('system_message', "You are a helpful assistant.")

        logger.info(f"System message: {self.system_message}")
        
        if not self.model:
            raise ValueError("vLLM requires model_path to be specified in config")
        
        self._initialize_vllm()
    
    def _get_tokenizer(self, llm, model_name):
        tokenizer = llm.get_tokenizer()
        if 'gemma-4' in model_name.lower():
            if tokenizer.chat_template is not None:
                return tokenizer

            # Workaround: Gemma-4 base models have no chat_template (https://github.com/huggingface/transformers/issues/45205)
            from huggingface_hub import hf_hub_download
            chat_template_path = hf_hub_download("google/gemma-4-E2B-it", "chat_template.jinja")
            with open(chat_template_path) as f:
                tokenizer.chat_template = f.read()

        return tokenizer

    def _initialize_vllm(self):
        try:
            logger.info(f"Initializing vLLM with model: {self.model}")
            
            self.llm = LLM(
                model=self.model,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_tokens,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True
            )
            
            self.tokenizer = self._get_tokenizer(self.llm, self.model)
            
            logger.info(f"✓ vLLM initialized successfully with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise
    
    def _format_prompt(self, prompt: str) -> str:
        if self.use_chat_template:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=ENABLE_THINKING
                )
                return formatted_prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
                return prompt
        return prompt
    
    def get_response(self, prompt: str, max_tokens: int = None) -> str:
        return self.get_responses([prompt], max_tokens)[0]
    
    def get_responses(self, prompts: List[str], max_tokens: int = None) -> List[str]:
        try:
            formatted_prompts = [self._format_prompt(p) for p in prompts]
            
            gen_max_tokens = max_tokens or self.config['train'].get('max_completion_length', 2048)

            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=gen_max_tokens,
                stop=self.stop_sequences,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                seed=self.seed
            )
            
            all_responses = []
            for i in range(0, len(formatted_prompts), self.batch_size):
                batch_prompts = formatted_prompts[i:i + self.batch_size]
                
                try:
                    outputs = self.llm.generate(batch_prompts, sampling_params)
                    
                    for output in outputs:
                        generated_text = output.outputs[0].text.strip()
                        all_responses.append(generated_text)
                        completion_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0
                        
                        self.total_output_tokens += completion_tokens
                        self.total_requests += 1
                except Exception as batch_e:
                    logger.error(f"vLLM batch generation failed: {batch_e}")
                    all_responses.extend([""] * len(batch_prompts))
                
                logger.debug(f"Generated batch {i//self.batch_size + 1}/{(len(formatted_prompts)-1)//self.batch_size + 1}")
            
            return all_responses
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            return [""] * len(prompts)

class LLMClientFactory:
    @staticmethod
    def create_client(config: Dict[str, Any], backend: str = None) -> BaseLLMClient:
        backend = backend or config.get('generator', {}).get('backend', 'api')
        
        if backend == 'api':
            return OpenAIClient(config)
        elif backend == 'vllm':
            return VLLMClient(config)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'api' or 'vllm'.")

class CoTGenerator:
    def __init__(self, config: Dict[str, Any], backend: str = None):
        generator_config = config.get('generator', {})

        self.config = config
        self.backend = backend or generator_config.get('backend', 'api')
        self.llm_client = LLMClientFactory.create_client(config, self.backend)
        self.should_validate_cot_steps = generator_config.get('validate_cot_steps', True)
        self.max_validation_attempts = generator_config.get('max_cot_validation_attempts', 3)
        self.max_generation_attempts = generator_config.get('max_generation_attempts', 3)
        self.min_cot_steps = generator_config.get('min_cot_steps', 1)
        self.max_cot_steps = generator_config.get('max_cot_steps', 200)
        self.dataset_configs = config.get('datasets', {})
        self.debug = config.get('debug', False)
        self.end_mark = END_MARK
        
        self.prerequisite_tokens = [name for name, cfg in COT_TOKENS.items() if cfg.get('prerequisite', False)]
        self.regular_tokens = [name for name in COT_TOKENS.keys() if name not in self.prerequisite_tokens]

        logger.info(f"Initialized CoTGenerator with backend: {self.backend}")
        if self.debug:
            logger.info(f"Prerequisite tokens: {self.prerequisite_tokens}")
            logger.info(f"Regular tokens: {self.regular_tokens}")
            logger.info(f"Max generation attempts: {self.max_generation_attempts}")
            logger.info(f"Min CoT steps: {self.min_cot_steps}")
            logger.info(f"Max CoT steps: {self.max_cot_steps}")
    
    def format_example(self, question: str, options: List[str] = None, cot_content: str = "") -> str:
        example = f"Question: {question}"
        if options:
            example += "\nOptions: "
            for i, opt in enumerate(options):
                example += f"{CHOICE_MAP[i]}. {opt}\n"
        
        if cot_content:
            example += f"Answer: {cot_content}\n\n"
        else:
            example += "\nAnswer: "
        
        return example

    def _generate_prerequisite_steps(self, question: str, answer: str) -> List[str]:
        if not self.prerequisite_tokens:
            return []

        token_descriptions = []
        for name in self.prerequisite_tokens:
            if self.end_mark:
                token_descriptions.append(f"<{name}> ... </{name}> : {COT_TOKENS[name]['description']}")
            else:
                token_descriptions.append(f"<{name}>: ... : {COT_TOKENS[name]['description']}")
        token_types_str = "\n".join(token_descriptions)

        if self.end_mark:
            format_instruction = "For each step, use exactly <token_name> content </token_name>."
        else:
            format_instruction = "For each step, use exactly <token_name>: content."

        prompt = f"""Generate ONLY the steps for the following token types. Do not output any other text, explanations, or commentary. Output each step on a new line.

Question: {question}

Token types to generate:
{token_types_str}

{format_instruction}

Now output the steps, one per line, starting immediately:
"""
        if self.debug:
            logger.info(f"Prerequisite prompt: {prompt}")

        try:
            response = self.llm_client.get_response(prompt)
            if not response or not response.strip():
                return []
            if self.debug:
                logger.info(f"Prerequisite response: {response[:1000]}...")
            steps = self.extract_labeled_content(response)
            filtered = []
            for step in steps:
                token_name, _ = self._parse_step(step)
                if token_name in self.prerequisite_tokens:
                    filtered.append(step)
            return filtered
        except Exception as e:
            logger.error(f"Error generating prerequisite steps: {e}")
            return []
        
    def planning_prompt(self, question: str, answer: str, prerequisite_steps: List[str] = None) -> str:
        token_descriptions = []
        for name in self.regular_tokens:
            if self.end_mark:
                token_descriptions.append(f"<{name}> ... </{name}> - {COT_TOKENS[name]['description']}")
            else:
                token_descriptions.append(f"<{name}>: ... - {COT_TOKENS[name]['description']}")
        token_types_str = "\n".join(token_descriptions)

        if self.end_mark:
            format_instruction = (
                "For labeled steps, use exactly <token_name> content </token_name>, "
                "where token_name must be one of the token types listed above."
            )
        else:
            format_instruction = (
                "For labeled steps, use exactly <token_name>: content, "
                "where token_name must be one of the token types listed above."
            )

        context = ""
        if prerequisite_steps:
            context = "Pre‑generated steps (use these as facts for the remaining reasoning):\n" + "\n".join(prerequisite_steps) + "\n\n"

        prompt = f"""Solve the question using step-by-step reasoning. Output each step on a new line. Do not output any extra text like "Step 1", "Thinking process", or explanations. Only the reasoning steps.

Question: {question}

{context}Available token types:
{token_types_str}

{format_instruction}

Steps that do not correspond to any token type can be plain text (unlabeled). However, at least one step must be labeled with a token from the list above.

Now output your reasoning steps, one per line, starting immediately:
"""
        return prompt

    def extract_labeled_content(self, input_string: str) -> List[str]:
        if not input_string or not input_string.strip():
            return []

        token_names_pattern = '|'.join(re.escape(name) for name in COT_TOKEN_NAMES)

        if not self.end_mark:
            lines = input_string.strip().split('\n')
            steps = []

            skip_patterns = [
                r'^\s*[\*\-]\s+',
                r'^\s*\d+\.\s+',
                r'(available token types|for labeled steps|critical rules|now output|pre‑generated steps|question:)'
            ]
            compiled_skip = re.compile('|'.join(skip_patterns), re.IGNORECASE)

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if compiled_skip.search(line):
                    continue

                token_match = re.match(rf'^<({token_names_pattern})>:\s*(.*)$', line, re.IGNORECASE)
                if token_match:
                    steps.append(line)
                else:
                    line = re.sub(r'\s+', ' ', line)
                    if line and line[0].islower():
                        line = line[0].upper() + line[1:]
                    if line and not any(line.endswith(p) for p in ['.', '!', '?', ':']):
                        if len(line.split()) > 2:
                            line += '.'
                    steps.append(line)

            seen = set()
            unique = []
            for s in steps:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            return unique

        pattern = rf'<({token_names_pattern})>\s*(.*?)\s*</\1>'
        flags = re.IGNORECASE | re.DOTALL
        matches = list(re.finditer(pattern, input_string, flags))

        steps = []
        pos = 0
        for match in matches:
            if pos < match.start():
                unlabeled = input_string[pos:match.start()].strip()
                if unlabeled:
                    unlabeled = re.sub(r'\s+', ' ', unlabeled)
                    if unlabeled and unlabeled[0].islower():
                        unlabeled = unlabeled[0].upper() + unlabeled[1:]
                    if unlabeled and not any(unlabeled.endswith(p) for p in ['.', '!', '?', ':']):
                        if len(unlabeled.split()) > 2:
                            unlabeled += '.'
                    steps.append(unlabeled)
            token_step = match.group(0).strip()
            if token_step:
                steps.append(token_step)
            pos = match.end()

        if pos < len(input_string):
            tail = input_string[pos:].strip()
            if tail:
                tail = re.sub(r'\s+', ' ', tail)
                if tail and tail[0].islower():
                    tail = tail[0].upper() + tail[1:]
                if tail and not any(tail.endswith(p) for p in ['.', '!', '?', ':']):
                    if len(tail.split()) > 2:
                        tail += '.'
                steps.append(tail)

        valid_token_pattern = rf'^\s*<({token_names_pattern})>\s*.*?\s*</\1>\s*$'
        filtered = []
        for step in steps:
            if '<' in step or '>' in step:
                if re.match(valid_token_pattern, step, re.DOTALL):
                    filtered.append(step)
            else:
                filtered.append(step)

        seen = set()
        unique = []
        for step in filtered:
            if step not in seen:
                seen.add(step)
                unique.append(step)
        return unique
    
    def _parse_step(self, step: str) -> Tuple[Optional[str], Optional[str]]:
        tag_pattern = rf'^\s*<({ "|".join(re.escape(n) for n in COT_TOKEN_NAMES) })>\s*(.*?)\s*</\1>\s*$'
        match = re.match(tag_pattern, step, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).lower(), match.group(2).strip() # Return (token_name, content)
        colon_pattern = rf'^\s*<({ "|".join(re.escape(n) for n in COT_TOKEN_NAMES) })>:\s*(.*?)\s*$'
        match = re.match(colon_pattern, step, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).lower(), match.group(2).strip()
        return None, None

    def validate_cot_steps(self, cot_steps: List[str]) -> bool:
        if not cot_steps:
            return False
        if not self.should_validate_cot_steps:
            return True

        single_token_mode = (len(COT_TOKEN_NAMES) == 1)
        min_steps = 1 if single_token_mode else self.min_cot_steps

        if len(cot_steps) > self.max_cot_steps or len(cot_steps) < min_steps:
            if self.debug:
                logger.debug(f"Validation failed: step count {len(cot_steps)} not in [{min_steps}, {self.max_cot_steps}]")
            return False

        token_step_counts = {name: 0 for name in COT_TOKEN_NAMES}
        seen_content = set()

        for step in cot_steps:
            token_name, content = self._parse_step(step)

            if ('<' in step or '>' in step) and token_name is None:
                if self.debug:
                    logger.debug(f"Validation failed: step contains angle brackets but is not a valid token step: {step[:50]}")
                return False

            if token_name is None:
                content = step.strip()
                if not content:
                    continue
                norm_content = content.lower().strip()
                if norm_content:
                    seen_content.add(norm_content)
                continue

            if content is None:
                if self.debug:
                    logger.debug(f"Validation failed: token step with missing content: {step[:50]}")
                return False

            if self._is_garbage_content(content):
                if self.debug:
                    logger.debug(f"Validation failed: garbage content in token step: {content[:50]}")
                return False

            norm_content = content.lower().strip()
            if norm_content:
                seen_content.add(norm_content)
            token_step_counts[token_name] += 1

        if len(seen_content) < 1:
            if self.debug:
                logger.debug(f"Validation failed: no unique content items")
            return False

        if len(COT_TOKEN_NAMES) > 1 and len(cot_steps) > 3:
            max_ratio = 0.85
            total = len(cot_steps)
            for count in token_step_counts.values():
                if count / total > max_ratio:
                    if self.debug:
                        logger.debug(f"Validation failed: token type ratio {count/total:.2f} exceeds {max_ratio}")
                    return False

        return True

    def is_high_quality_cot(self, cot_steps: List[str]) -> bool:
        if not cot_steps:
            return False

        single_token_mode = (len(COT_TOKEN_NAMES) == 1)
        min_steps = 1 if single_token_mode else self.min_cot_steps
        if len(cot_steps) < min_steps:
            return False

        total_content_length = 0
        has_cot_token = False

        for step in cot_steps:
            token_name, content = self._parse_step(step)

            if token_name is None and single_token_mode:
                content = step.strip()
                if not content:
                    return False
                total_content_length += len(content)
                continue

            if token_name is None or content is None:
                return False

            total_content_length += len(content)
            if token_name in COT_TOKENS:
                has_cot_token = True

        avg_content_length = total_content_length / len(cot_steps)
        if avg_content_length < 20:
            return False
        if not has_cot_token and not single_token_mode:
            return False
        return True

    def _is_garbage_content(self, content: str) -> bool:
        if not content:
            return True
        
        content_lower = content.lower().strip()
        
        garbage_patterns = [
            r'^\s*s\s*[:.]',
            r'^\s*ing\.',
            r'^\s*steps?\.',
            r'^\s*wait,',
            r'^\s*but\s+',
            r'^\s*so\s+',
            r'^\s*well,',
            r'^\s*</?think>',
            r'^\s*\*+\s*$',
            r'^\s*\.\.+',
            r'^\s*$',
            r'^\s*then\s+the\s*$',
            r'^\s*and\s+then\s*$',
            r'^\s*extract\s+',
            r'^\s*state\s+',
            r'^\s*list\s+',
            r'^\s*first\s+',
            r'^\s*next\s+',
            r'^\s*i\s+',
            r'^\s*we\s+'
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, content_lower):
                return True
        
        if len(content_lower) < 4:
            return True
        
        words = content_lower.split()
        if len(words) <= 2:
            if not any(word.isdigit() for word in words):
                return True
        
        cutoff_endings = [' a', ' an', ' the', ' to', ' of', ' for', ' with', ' without', ' and', ' or', ' but', ' then', ' so']
        for ending in cutoff_endings:
            if content_lower.endswith(ending):
                return True
        
        return False

    def get_cot_steps(self, question: str, answer: str) -> List[str]:
        if self.debug:
            logger.info(f"Generating CoT for question: {question[:100]}...")

        prereq_steps = self._generate_prerequisite_steps(question, answer)
        if self.debug:
            logger.info(f"Generated {len(prereq_steps)} prerequisite steps")

        if self.prerequisite_tokens and not prereq_steps:
            if self.debug:
                logger.info("No prerequisite steps generated, aborting generation")
            return []

        if not self.regular_tokens:
            if not prereq_steps:
                return []
            if not self.validate_cot_steps(prereq_steps):
                if self.debug:
                    logger.info("Prerequisite steps failed validation")
                return [] if self.should_validate_cot_steps else prereq_steps
            return prereq_steps

        planning_prompt = self.planning_prompt(question, answer, prerequisite_steps=prereq_steps)

        if self.debug:
            logger.info(f"Planning prompt: {planning_prompt}")

        try:
            plan_response = self.llm_client.get_response(planning_prompt)
            if not plan_response or plan_response.strip() == "":
                if self.debug:
                    logger.info(f"Empty plan response for question: {question[:50]}...")
                return []
            if self.debug:
                logger.info(f"Plan response (first 1000 chars): {plan_response[:1000]}...")
            if self.debug:
                logger.info(f"Plan response length: {len(plan_response)} characters")
        except Exception as e:
            logger.error(f"Error getting planning response: {e}")
            return []

        reasoning_steps = self.extract_labeled_content(plan_response)

        if self.debug:
            logger.info(f"Raw planning response:\n{plan_response[:1000]}")
            logger.info(f"Extracted steps: {reasoning_steps}")
            logger.info(f"Validation passed: {self.validate_cot_steps(reasoning_steps)}")

        if self.regular_tokens and not any('<' in step or '>' in step for step in reasoning_steps):
            if self.debug:
                logger.info("Reasoning steps contain no labeled step; rejecting generation")
            return []

        cot_steps = prereq_steps + reasoning_steps
        
        if not cot_steps:
            if self.debug:
                logger.info("No CoT steps extracted")
            return []

        if not self.validate_cot_steps(cot_steps):
            if self.debug:
                logger.info("CoT steps failed validation")
            if not self.should_validate_cot_steps:
                return cot_steps
            else:
                return []

        return cot_steps
    
    def get_cot_steps_with_retry(self, question: str, answer: str, max_attempts: int = None) -> List[str]:
        max_attempts = max_attempts or self.max_generation_attempts
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    if self.debug:
                        logger.info(f"Retry attempt {attempt + 1}/{max_attempts} for question: {question[:50]}...")
                
                cot_steps = self.get_cot_steps(question, answer)
                
                if cot_steps:
                    if self.debug:
                        logger.info(f"Successfully generated {len(cot_steps)} CoT steps on attempt {attempt + 1}")
                    return cot_steps
                else:
                    if self.debug:
                        logger.info(f"Empty CoT steps on attempt {attempt + 1}/{max_attempts}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                
                if attempt == max_attempts - 1:
                    logger.error(f"Failed to generate CoT after {max_attempts} attempts")
                    return []
        
        return []
    
    def batch_generate_cot_steps(self, questions: List[str], answers: List[str]) -> List[List[str]]:
        if len(questions) != len(answers):
            raise ValueError(f"Number of questions ({len(questions)}) doesn't match number of answers ({len(answers)})")

        all_prereq_steps = []
        example_valid = []
        for q, a in zip(questions, answers):
            prereq = self._generate_prerequisite_steps(q, a)
            all_prereq_steps.append(prereq)
            if self.prerequisite_tokens and not prereq:
                example_valid.append(False)
            else:
                example_valid.append(True)

        if not self.regular_tokens:
            all_cot_steps = []
            validation_stats = {"valid": 0, "invalid": 0, "empty": 0}
            for i, is_valid in enumerate(example_valid):
                if not is_valid:
                    all_cot_steps.append([])
                    validation_stats["invalid"] += 1
                    continue
                cot_steps = all_prereq_steps[i]
                if not cot_steps:
                    all_cot_steps.append([])
                    validation_stats["empty"] += 1
                    continue
                if self.should_validate_cot_steps:
                    is_valid_steps = self.validate_cot_steps(cot_steps)
                    if is_valid_steps:
                        all_cot_steps.append(cot_steps)
                        validation_stats["valid"] += 1
                    else:
                        all_cot_steps.append([])
                        validation_stats["invalid"] += 1
                else:
                    all_cot_steps.append(cot_steps)
                    validation_stats["valid"] += 1
            logger.info(f"CoT generation stats: {validation_stats['valid']} valid, {validation_stats['invalid']} invalid, {validation_stats['empty']} empty")
            return all_cot_steps

        all_planning_prompts = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            if not example_valid[i]:
                all_planning_prompts.append(None)
            else:
                all_planning_prompts.append(self.planning_prompt(q, a, prerequisite_steps=all_prereq_steps[i]))

        prompts_to_gen = [p for p in all_planning_prompts if p is not None]
        if prompts_to_gen:
            plan_responses = self.llm_client.get_responses(prompts_to_gen)
        else:
            plan_responses = []

        all_cot_steps = []
        validation_stats = {"valid": 0, "invalid": 0, "empty": 0}
        resp_idx = 0

        for i, is_valid in enumerate(example_valid):
            if not is_valid:
                all_cot_steps.append([])
                validation_stats["invalid"] += 1
                continue

            plan_response = plan_responses[resp_idx]
            resp_idx += 1

            if not plan_response or plan_response.strip() == "":
                all_cot_steps.append([])
                validation_stats["empty"] += 1
                continue

            reasoning_steps = self.extract_labeled_content(plan_response)

            if self.regular_tokens and not any('<' in step or '>' in step for step in reasoning_steps):
                all_cot_steps.append([])
                validation_stats["invalid"] += 1
                continue

            cot_steps = all_prereq_steps[i] + reasoning_steps

            if not cot_steps:
                all_cot_steps.append([])
                validation_stats["empty"] += 1
                continue

            if len(cot_steps) < self.min_cot_steps:
                all_cot_steps.append([])
                validation_stats["invalid"] += 1
                continue

            if self.should_validate_cot_steps:
                is_valid_steps = self.validate_cot_steps(cot_steps)
                if is_valid_steps:
                    all_cot_steps.append(cot_steps)
                    validation_stats["valid"] += 1
                else:
                    all_cot_steps.append([])
                    validation_stats["invalid"] += 1
            else:
                all_cot_steps.append(cot_steps)
                validation_stats["valid"] += 1

        logger.info(f"CoT generation stats: {validation_stats['valid']} valid, {validation_stats['invalid']} invalid, {validation_stats['empty']} empty")
        return all_cot_steps
    
    def generate_with_quality_check(self, question: str, answer: str, max_attempts: int = None) -> List[str]:
        max_attempts = max_attempts or self.max_generation_attempts
        
        for attempt in range(max_attempts):
            cot_steps = self.get_cot_steps_with_retry(question, answer, max_attempts=1)
            
            if self.is_high_quality_cot(cot_steps):
                if self.debug:
                    logger.info(f"Generated high-quality CoT with {len(cot_steps)} steps on attempt {attempt + 1}")
                return cot_steps
            
            if attempt < max_attempts - 1:
                if self.debug:
                    logger.info(f"CoT quality insufficient, retrying... (attempt {attempt + 1}/{max_attempts})")
        
        if self.debug:
            logger.info(f"Failed to generate high-quality CoT after {max_attempts} attempts")
        return []

class DatasetGenerator:
    ANSWER_TYPE_MULTIPLE_CHOICE = "multiple_choice"
    ANSWER_TYPE_BOOLEAN = "boolean"
    ANSWER_TYPE_NUMERIC = "numeric"
    ANSWER_TYPE_OPEN_ENDED = "open_ended"
    
    def __init__(
        self,
        cot_generator: CoTGenerator,
        config: Dict[str, Any],
        args: Optional[GeneratorArguments]
    ):
        dataset_name = args.dataset
        datasets_config = load_datasets_config()

        self.args = args
        self.config = config
        self.dataset_name = dataset_name
        self.cot_generator = cot_generator
        self.generator_config = config['generator']
        self.dataset_config = datasets_config[dataset_name]
        
        self.output_dir = Path(self.generator_config.get('output_dir', DEFAULT_DATA_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.incremental_save = self.generator_config.get('incremental_save', True)
        self.filter_output = self.generator_config.get('filter_output', True)
        self.incremental_save_interval = self.generator_config.get('incremental_save_interval', 1000)
        
        self.max_retries = self.cot_generator.llm_client.max_retries
        self.batch_size = self.generator_config.get('batch_size', self.cot_generator.llm_client.batch_size)
        self.debug = self.cot_generator.debug

        peft_mode = config.get('common', {}).get('parameter_efficient_mode', '')
        self.use_cot_tokens = peft_mode in ('lora-cog-frozen', 'lora-cog-tuned')
        
        self._initialize_configuration()
        
        logger.info(f"Dataset {dataset_name}: answer_type={self._answer_type}")
        logger.info(f"Valid answers: {self._valid_answers}")
        logger.info(f"Field mapping: {self._field_mapping}")
        logger.info(f"LaTeX cleaning: {'Enabled' if self._clean_latex_enabled else 'Disabled'}")
    
    def _initialize_configuration(self):
        self._answer_type = self.get_answer_type()
        self._valid_answers = self.get_valid_answers()
        self._choice_labels = self.get_choice_labels()
        self._field_mapping = self.get_field_mapping()
        
        self._clean_latex_enabled = self.dataset_config.get('clean_latex', False)
        self._additional_config = self.dataset_config.get('additional_config', {})
    
    def get_answer_type(self) -> str:
        return self.dataset_config['answer_type']
    
    def get_valid_answers(self) -> Optional[Set[str]]:
        if self.get_answer_type() != self.ANSWER_TYPE_MULTIPLE_CHOICE:
            return None
        
        valid_answers = self.dataset_config.get('valid_answers')
        if valid_answers is None:
            return None
        
        if isinstance(valid_answers, list):
            return set(str(v).upper() for v in valid_answers)
        elif isinstance(valid_answers, str):
            if '-' in valid_answers:
                # Range like "A-D"
                start, end = valid_answers.split('-')
                if len(start) == 1 and len(end) == 1:
                    return set(chr(c) for c in range(ord(start.upper()), ord(end.upper()) + 1))
            return set(v.strip().upper() for v in valid_answers.split(','))
        
        return None
    
    def get_choice_labels(self) -> List[str]:
        choice_labels = self.dataset_config.get('choice_labels')
        if choice_labels:
            if isinstance(choice_labels, list):
                return [str(label).upper() for label in choice_labels]
            elif isinstance(choice_labels, str):
                return [label.strip().upper() for label in choice_labels.split(',')]
        
        valid_answers = self.get_valid_answers()
        if valid_answers:
            valid_sorted = sorted(valid_answers, key=lambda x: x.upper())
            return [label.upper() for label in valid_sorted]
        
        return ['A', 'B', 'C', 'D', 'E']
    
    def get_field_mapping(self) -> Dict[str, str]:
        config_mapping = self.dataset_config.get('field_mapping', {})
        
        default_mapping = {
            'question': 'question',
            'answer': 'answer',
            'options': 'options',
            'choices': 'choices'
        }
        
        merged = default_mapping.copy()
        merged.update(config_mapping)
        
        return merged
    
    def extract_answer_from_example(self, example: Dict) -> Optional[Any]:
        return None
    
    def parse_options(self, options: Any) -> Optional[List[str]]:
        return None
    
    def get_answer_keywords(self) -> List[str]:
        return ['answer', 'option', 'choice', 'correct', 'solution', 'result', 'final']
    
    def get_answer_prefixes(self) -> List[str]:
        return ['Answer:', 'Solution:', 'Response:', 'The answer is:', 'Result:']
    
    def _clean_latex(self, text: str) -> str:
        if not text:
            return text
        
        text = text.replace('\\(', '').replace('\\)', '')
        text = text.replace('\\[', '').replace('\\]', '')
        
        text = re.sub(r'\$(.*?)\$', r'\1', text)
        text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
        
        text = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'(\1)/(\2)', text, flags=re.DOTALL)
        text = re.sub(r'\\sqrt{([^}]+)}', r'sqrt(\1)', text)
        text = re.sub(r'\\sqrt\[([^\]]+)\]{(.+?)}', r'(\2)^(1/\1)', text, flags=re.DOTALL)
        
        text = re.sub(r'\\text{(.+?)}', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\\mathrm{(.+?)}', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\\mathbf{(.+?)}', r'\1', text, flags=re.DOTALL)
        
        greek_map = {
            r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\Gamma': 'Γ',
            r'\\delta': 'δ', r'\\Delta': 'Δ', r'\\epsilon': 'ε', r'\\varepsilon': 'ε',
            r'\\zeta': 'ζ', r'\\eta': 'η', r'\\theta': 'θ', r'\\Theta': 'Θ',
            r'\\iota': 'ι', r'\\kappa': 'κ', r'\\lambda': 'λ', r'\\Lambda': 'Λ',
            r'\\mu': 'μ', r'\\nu': 'ν', r'\\xi': 'ξ', r'\\Xi': 'Ξ',
            r'\\pi': 'π', r'\\Pi': 'Π', r'\\rho': 'ρ', r'\\sigma': 'σ',
            r'\\Sigma': 'Σ', r'\\tau': 'τ', r'\\upsilon': 'υ', r'\\Upsilon': 'Υ',
            r'\\phi': 'φ', r'\\Phi': 'Φ', r'\\chi': 'χ', r'\\psi': 'ψ',
            r'\\Psi': 'Ψ', r'\\omega': 'ω', r'\\Omega': 'Ω'
        }
        
        for latex, unicode in greek_map.items():
            text = text.replace(latex, unicode)
        
        operator_map = {
            r'\\times': '×', r'\\cdot': '·', r'\\div': '÷',
            r'\\pm': '±', r'\\mp': '∓', r'\\leq': '≤', r'\\le': '≤',
            r'\\geq': '≥', r'\\ge': '≥', r'\\neq': '≠', r'\\ne': '≠',
            r'\\approx': '≈', r'\\sim': '∼', r'\\propto': '∝',
            r'\\infty': '∞', r'\\partial': '∂', r'\\nabla': '∇'
        }
        
        for latex, unicode in operator_map.items():
            text = re.sub(latex, unicode, text)
        
        text = re.sub(r'\^{([^}]+)}', r'^\1', text)
        text = re.sub(r'_{([^}]+)}', r'_\1', text)
        
        text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_dataset(self, split: str):
        source = self.dataset_config.get('source')
        if not source:
            logger.error(f"No source specified for dataset {self.dataset_name}")
            raise ValueError(f"Dataset source not configured for {self.dataset_name}")
        
        split_mapping = self.dataset_config.get('split_mapping', {})
        config_name = self.dataset_config.get('config_name')
        actual_split = split_mapping.get(split, split)
        
        try:
            from datasets import load_dataset
            logger.info(f"Loading dataset: {source}, split: {actual_split}")
            
            if config_name:
                dataset = load_dataset(source, config_name)
            else:
                dataset = load_dataset(source)
            
            if actual_split not in dataset:
                available_splits = list(dataset.keys())
                logger.error(f"Split '{actual_split}' not found. Available splits: {available_splits}")
                raise ValueError(f"Split '{actual_split}' not found in dataset")
            
            logger.info(f"Successfully loaded {len(dataset[actual_split])} examples from {source}/{actual_split}")
            return dataset[actual_split]
        except ImportError:
            logger.error("The 'datasets' library is not installed. Please install it with: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset {source} split {actual_split}: {e}")
            raise
    
    def get_output_filename(self, split: str) -> Path:
        if self.dataset_name:
            base_name = f"{self.dataset_name}_{split}"
        else:
            dataset_name = self.__class__.__name__.replace('Generator', '')
            base_name = f"{dataset_name}_{split}"
        
        base_name = base_name.lower()
        
        return self.output_dir / f"{base_name}.json"
    
    def save_results(self, results: List[Dict], output_file: Path):
        try:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved {len(results)} examples to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            raise
    
    def filter_results(self, results: List[Dict], split: str) -> List[Dict]:
        cleaned = []

        meta_patterns = [
            r'\bthinking process\b',
            r'\banalyze the request\b',
            r'\bconstraint check\b',
            r'\bdrafting\b',
            r'\boutput only\b',
            r'\beach step must be\b',
            r'\bline\s*\d+\b',
            r'\bneed to check\b',
            r'\bokay\b',
            r'\bnote:\b'
        ]
        meta_re = re.compile('|'.join(meta_patterns), re.IGNORECASE)

        # Placeholder content patterns for token steps (excluding "<content>")
        placeholder_patterns = [
            r'^\.\.\.$',
            r'^content$',
            r'^text$'
        ]
        placeholder_re = re.compile('|'.join(placeholder_patterns), re.IGNORECASE)

        for item in results:
            if not item:
                continue
            cot_steps = item.get('cot_steps', [])
            if split == "test":
                cleaned.append(item)
                continue

            filtered = []
            for step in cot_steps:
                step = step.strip()
                if not step:
                    continue

                if len(step) < 3:
                    continue
                if re.match(r'^[\d\s\.,!?;:-]+$', step):
                    continue
                if '`' in step or '*' in step:
                    continue

                if meta_re.search(step):
                    if self.debug:
                        logger.debug(f"Skipping step due to meta-pattern: {step[:50]}")
                    continue

                if '<' in step or '>' in step:
                    token_name, content = self.cot_generator._parse_step(step)
                    if token_name is None or content is None:
                        continue
                    if placeholder_re.match(content.strip()):
                        continue
                    if len(content.strip()) < 3:
                        continue

                filtered.append(step)

            if not filtered:
                continue

            cleaned.append({
                "question": item.get('question', ''),
                "answer": item.get('answer', ''),
                "cot_steps": filtered,
                "split": item.get('split', split)
            })

        return cleaned
    
    def format_question_and_answer(self, example: Dict) -> Tuple[str, str]:
        return self._format_question_and_answer_internal(example)
    
    def _format_question_and_answer_internal(self, example: Dict) -> Tuple[str, str]:
        try:
            raw_question = self._get_field(example, 'question')
            if not raw_question:
                logger.warning(f"No question found in example: {example.keys()}")
                raise ValueError("No question found")
            
            raw_answer = None
            
            if hasattr(self, 'extract_answer_from_example'):
                try:
                    raw_answer = self.extract_answer_from_example(example)
                except Exception as e:
                    logger.debug(f"Custom answer extraction failed: {e}")
            
            if raw_answer is None:
                raw_answer = self._get_field(example, 'answer')
            
            if raw_answer is None:
                logger.warning(f"No answer found in example: {example.keys()}")
                raise ValueError("No answer found")
            
            raw_options = self._get_field(example, 'options')
            if raw_options is None:
                raw_options = self._get_field(example, 'choices')
            
            question = self._format_question_with_options(raw_question, raw_options)
            
            if self._clean_latex_enabled:
                question = self._clean_latex(question)
            
            answer = self._normalize_answer_internal(raw_answer, raw_options)
            
            return question, answer
        except Exception as e:
            logger.warning(f"Error formatting example: {e}")
            raise
    
    def _get_field(self, example: Dict, field_name: str) -> Any:
        mapped_field = self._field_mapping.get(field_name)
        if mapped_field and mapped_field in example:
            return example[mapped_field]
        
        if field_name in example:
            return example[field_name]
        
        alternatives = self._get_field_alternatives(field_name)
        for alt in alternatives:
            if alt in example:
                return example[alt]
        
        return None
    
    def _get_field_alternatives(self, field_name: str) -> List[str]:
        if field_name == 'question':
            return ['query', 'problem', 'prompt', 'text', 'sentence', 'content', 'stem']
        elif field_name == 'answer':
            return ['correct', 'correct_answer', 'solution', 'response',
                   'label', 'target', 'gold_answer', 'answer_key']
        elif field_name in ['options', 'choices']:
            return ['choices', 'options', 'candidates', 'alternatives']
        return []
    
    def _format_question_with_options(self, question: str, options: Any) -> str:
        question = str(question).strip()
        
        question = self._clean_question(question)
        
        if not options:
            return question
        
        parsed_options = None
        if hasattr(self, 'parse_options'):
            parsed_options = self.parse_options(options)
        
        if parsed_options is None:
            parsed_options = self._parse_options_internal(options)
        
        if not parsed_options:
            return question
        
        formatted_options = self._format_options(parsed_options)
        
        return f"{question}\n\n{formatted_options}"
    
    def _clean_question(self, question: str) -> str:
        question = str(question).strip()
        
        prefixes = ['Question:', 'Problem:', 'Q:', 'Prompt:']
        for prefix in prefixes:
            if question.startswith(prefix):
                question = question[len(prefix):].strip()
                break
        
        if self._clean_latex_enabled:
            question = self._clean_latex(question)
        
        if not question.endswith(('?', '!', '.')):
            if any(word in question.lower() for word in ['what', 'why', 'how', 'when', 'where', 'which']):
                question += '?'
            else:
                question += '.'
        
        return question
    
    def _parse_options_internal(self, options: Any) -> List[str]:
        if not options:
            return []
        
        if isinstance(options, list):
            return [str(opt).strip() for opt in options]
        
        if isinstance(options, str):
            options = options.strip()
            
            if options.startswith('[') and options.endswith(']'):
                try:
                    parsed = json.loads(options)
                    if isinstance(parsed, list):
                        return [str(opt).strip() for opt in parsed]
                except:
                    pass
            
            if ',' in options:
                import shlex
                try:
                    lexer = shlex.shlex(options, posix=True)
                    lexer.whitespace = ','
                    lexer.whitespace_split = True
                    parts = list(lexer)
                    return [part.strip().strip("'\"") for part in parts]
                except:
                    return [opt.strip().strip("'\"") for opt in options.split(',')]
            
            return [options]
        
        return [str(options)]
    
    def _format_options(self, options: List[str]) -> str:
        if not options:
            return ""
        
        formatted = []
        for i, option in enumerate(options):
            if i >= len(self._choice_labels):
                break
            label = self._choice_labels[i]
            
            option_text = str(option).strip()
            option_text = re.sub(r'^[A-Z][).:]\s*', '', option_text)
            option_text = re.sub(r'^\([A-Z]\)\s*', '', option_text)
            
            if self._clean_latex_enabled:
                option_text = self._clean_latex(option_text)
            
            formatted.append(f"{label}. {option_text}")
        
        return "\n".join(formatted)
    
    def _normalize_answer_internal(self, raw_answer: Any, options: Any = None) -> str:
        answer_type = self._answer_type
        
        if answer_type == self.ANSWER_TYPE_MULTIPLE_CHOICE:
            return self._normalize_multiple_choice_answer(raw_answer, options)
        elif answer_type == self.ANSWER_TYPE_BOOLEAN:
            return self._normalize_boolean_answer(raw_answer)
        elif answer_type == self.ANSWER_TYPE_NUMERIC:
            return self._normalize_numeric_answer(raw_answer)
        else:
            return self._normalize_open_ended_answer(raw_answer)
    
    def _normalize_multiple_choice_answer(self, raw_answer: Any, options: Any) -> str:
        if not self._valid_answers:
            logger.warning("No valid answers defined for multiple choice dataset")
            return self._fallback_normalize_answer(raw_answer, options)
        
        answer_str = str(raw_answer).strip().upper()
        
        valid_letters = ''.join(sorted(self._valid_answers))
        letter_match = re.search(rf'^[{valid_letters}]$', answer_str)
        if letter_match:
            return letter_match.group(0)
        
        patterns = self._get_multiple_choice_patterns()
        
        for pattern in patterns:
            match = re.search(pattern, answer_str, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in self._valid_answers:
                    return letter
        
        if answer_str.isdigit():
            try:
                idx = int(answer_str) - 1
                valid_list = sorted(self._valid_answers)
                if 0 <= idx < len(valid_list):
                    return valid_list[idx]
            except:
                pass
        
        if options:
            parsed_options = self._parse_options_internal(options)
            for i, option in enumerate(parsed_options):
                option_lower = option.lower()
                answer_lower = answer_str.lower()
                if answer_lower in option_lower or option_lower in answer_lower:
                    valid_list = sorted(self._valid_answers)
                    if i < len(valid_list):
                        return valid_list[i]
        
        valid_list = sorted(self._valid_answers)
        return valid_list[0] if valid_list else 'A'
    
    def _get_multiple_choice_patterns(self) -> List[str]:
        valid_letters = ''.join(sorted(self._valid_answers)) if self._valid_answers else 'ABCDE'
        
        return [
            rf'^OPTION\s*[{valid_letters}]',
            rf'^CHOICE\s*[{valid_letters}]',
            rf'^ANSWER\s*[:\-]?\s*[{valid_letters}]',
            rf'^THE ANSWER IS\s*[:\-]?\s*[{valid_letters}]'
        ]
    
    def _fallback_normalize_answer(self, raw_answer: Any, options: Any) -> str:
        answer_str = str(raw_answer).strip().upper()
        
        patterns = self._get_multiple_choice_patterns()
        
        for pattern in patterns:
            match = re.search(pattern, answer_str, re.IGNORECASE)
            if match:
                return match.group(0)[-1].upper()
        
        if re.match(r'^[A-E]$', answer_str):
            return answer_str
        
        if answer_str.isdigit():
            try:
                idx = int(answer_str) - 1
                if 0 <= idx < len(self._choice_labels):
                    return self._choice_labels[idx]
            except:
                pass
        
        return self._choice_labels[0] if self._choice_labels else 'A'
    
    def _normalize_boolean_answer(self, raw_answer: Any) -> str:
        if isinstance(raw_answer, bool):
            return "true" if raw_answer else "false"
        
        answer_str = str(raw_answer).strip().lower()
        
        true_values = {'true', 'yes', 'correct', 'right', '1', 't', 'y'}
        false_values = {'false', 'no', 'incorrect', 'wrong', '0', 'f', 'n'}
        
        if answer_str in true_values:
            return "true"
        elif answer_str in false_values:
            return "false"
        
        patterns = [
            r'the answer is\s*[:\-]?\s*(true|false|yes|no)',
            r'answer\s*[:\-]?\s*(true|false|yes|no)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer_str, re.IGNORECASE)
            if match:
                extracted = match.group(1).lower()
                if extracted in ['true', 'yes']:
                    return "true"
                elif extracted in ['false', 'no']:
                    return "false"
        
        return "true"
    
    def _normalize_numeric_answer(self, raw_answer: Any) -> str:
        answer_str = str(raw_answer).strip()
        
        if self._clean_latex_enabled:
            answer_str = self._clean_latex(answer_str)
        
        # Extract from GSM8K format
        if "####" in answer_str:
            answer_str = answer_str.split("####")[-1].strip()
        
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_str.replace(',', ''))
        if numbers:
            return numbers[-1]
        
        patterns = [
            r'the answer is\s*[:\-]?\s*([\d\.]+)',
            r'answer\s*[:\-]?\s*([\d\.]+)',
            r'result\s*[:\-]?\s*([\d\.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer_str, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return answer_str
    
    def _normalize_open_ended_answer(self, raw_answer: Any) -> str:
        answer_str = str(raw_answer).strip()
        
        if self._clean_latex_enabled:
            answer_str = self._clean_latex(answer_str)
        
        prefixes = self.get_answer_prefixes()
        for prefix in prefixes:
            if answer_str.lower().startswith(prefix.lower()):
                answer_str = answer_str[len(prefix):].strip()
                break
        
        return answer_str
    
    def process_example(self, example: Dict, split: str) -> Dict:
        question, answer = self._format_question_and_answer_internal(example)
        
        cot_steps = []
        if split != "test" and self.use_cot_tokens:
            for retry in range(self.max_retries):
                try:
                    cot_steps = self.cot_generator.get_cot_steps_with_retry(question, answer)
                    if cot_steps:
                        break
                    logger.warning(f"Empty CoT steps for question: {question[:50]}... Attempt {retry + 1}/{self.max_retries}")
                except Exception as e:
                    logger.error(f"Error generating CoT for question: {e}")
                    if retry == self.max_retries - 1:
                        logger.error(f"Failed after {self.max_retries} retries")
        
        if self.use_cot_tokens and cot_steps and hasattr(self, '_postprocess_cot_steps'):
            cot_steps = self._postprocess_cot_steps(cot_steps, answer)
        
        return {
            "question": question,
            "answer": answer,
            "cot_steps": cot_steps,
            "split": split
        }
    
    def process_batch(self, examples: List[Dict], split: str) -> List[Dict]:
        if split == "test" or not self.use_cot_tokens:
            results = []
            for example in examples:
                question, answer = self._format_question_and_answer_internal(example)
                results.append({
                    "question": question,
                    "answer": answer,
                    "cot_steps": [],
                    "split": split
                })
            return results
        
        questions = []
        answers = []
        example_objs = []
        
        for example in examples:
            try:
                question, answer = self._format_question_and_answer_internal(example)
                questions.append(question)
                answers.append(answer)
                example_objs.append((question, answer, example))
            except Exception as e:
                logger.warning(f"Failed to process example: {example}")
                continue
        
        logger.info(f"Batch generating CoT steps for {len(questions)} examples")
        
        try:
            all_cot_steps = self.cot_generator.batch_generate_cot_steps(questions, answers)
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return [self.process_example(example, split) for example in examples]
        
        results = []
        for (question, answer, example), cot_steps in zip(example_objs, all_cot_steps):
            if cot_steps and hasattr(self, '_postprocess_cot_steps'):
                cot_steps = self._postprocess_cot_steps(cot_steps, answer)
            
            results.append({
                "question": question,
                "answer": answer,
                "cot_steps": cot_steps,
                "split": split
            })
        
        return results
    
    def _should_use_batch_processing(self, split: str, backend: str) -> bool:
        return (backend == 'vllm' and self.batch_size > 1 and split != "test")
    
    def _generate_for_split(self, data, split: str, num_examples: Optional[int] = None) -> List[Dict]:
        backend = self.cot_generator.config.get('generator', {}).get('backend', 'api')
        output_file = self.get_output_filename(split)
        results = []

        data_list = list(data)
        if num_examples and len(data_list) > num_examples:
            indices = random.sample(range(len(data_list)), num_examples)
            data_list = [data_list[i] for i in indices]
            logger.info(f"Sampled {num_examples} examples for {split} split")

        if split == "test":
            logger.info(f"Generating test data without CoT steps")
            for i, example in enumerate(tqdm(data_list, desc=f"Processing {self.dataset_name} {split}")):
                try:
                    question, answer = self._format_question_and_answer_internal(example)
                    results.append({
                        "question": question,
                        "answer": answer,
                        "cot_steps": [],
                        "split": split
                    })
                    if self.incremental_save and (i + 1) % self.incremental_save_interval == 0:
                        self.save_results(results, output_file)
                except Exception as e:
                    logger.warning(f"Failed to process test example {i}: {e}")
                    continue
        else:
            use_batch = self._should_use_batch_processing(split, backend)

            if use_batch:
                logger.info(f"Using batch processing with batch_size={self.batch_size}")
                for i in range(0, len(data_list), self.batch_size):
                    batch_data = data_list[i:i + self.batch_size]
                    batch_num = i // self.batch_size + 1
                    total_batches = (len(data_list) + self.batch_size - 1) // self.batch_size
                    logger.info(f"Processing batch {batch_num}/{total_batches}")

                    try:
                        batch_results = self.process_batch(batch_data, split)
                        for res in batch_results:
                            if not self.use_cot_tokens or res.get('cot_steps'):
                                results.append(res)
                        if self.incremental_save and (i + 1) % self.incremental_save_interval == 0:
                            self.save_results(results, output_file)
                    except Exception as e:
                        logger.error(f"Failed to process batch starting at {i}: {e}")
                        for example in batch_data:
                            try:
                                processed = self.process_example(example, split)
                                if not self.use_cot_tokens or processed.get('cot_steps'):
                                    results.append(processed)
                            except Exception as e2:
                                logger.warning(f"Failed to process individual example: {e2}")
                                continue
            else:
                logger.info(f"Processing {split} examples individually")
                for i, example in enumerate(tqdm(data_list, desc=f"Processing {self.dataset_name} {split}")):
                    try:
                        processed = self.process_example(example, split)
                        if not self.use_cot_tokens or processed.get('cot_steps'):
                            results.append(processed)
                        if self.incremental_save and (i + 1) % self.incremental_save_interval == 0:
                            self.save_results(results, output_file)
                    except Exception as e:
                        logger.warning(f"Failed to process example {i}: {e}")
                        continue

        self.save_results(results, output_file)

        if self.filter_output and split != "test" and self.use_cot_tokens:
            logger.info(f"Filtering {split} results")
            results = self.filter_results(results, split)
            output_file = self.get_output_filename(split)
            self.save_results(results, output_file)

        logger.info(f"Generated {len(results)} examples for {self.dataset_name} {split}")
        return results
    
    def generate(self, split: str, num_examples: Optional[int] = None) -> List[Dict]:
        try:
            data = self.get_dataset(split)
        except Exception as e:
            logger.error(f"Failed to load {self.dataset_name} dataset: {e}")
            return []
        
        return self._generate_for_split(data, split, num_examples)

class InteractiveGenerator(ABC):
    def __init__(
        self,
        config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        args: Optional[GeneratorArguments],
        show_action=SHOW_ACTION,
        show_prompt=SHOW_PROMPT,
        show_response=SHOW_RESPONSE,
        show_trajectory_on_fail=SHOW_TRAJECTORY_ON_FAIL
    ):
        self.config = config
        self.dataset_config = dataset_config
        self.args = args
        self.dataset_name = args.dataset
        self.mode = args.mode if args else 'train'
        self.show_action = show_action
        self.show_prompt = show_prompt
        self.show_response = show_response
        self.show_trajectory_on_fail = show_trajectory_on_fail

        self.task_source = dataset_config.get("task_source", "put_two")
        self.output_dir = Path(dataset_config.get("output_dir", DEFAULT_DATA_DIR))
        self.max_steps = dataset_config.get("max_steps", 50)
        self.output_format = dataset_config.get("output_format", "messages")
        self.filter_success = dataset_config.get("filter_success", True)
        self.fallback_policy: Optional[Callable] = None

        peft_mode = config.get('common', {}).get('parameter_efficient_mode', '')
        self.use_cot_tokens = peft_mode in ('lora-cog-frozen', 'lora-cog-tuned')
        if self.use_cot_tokens:
            logger.info("CoT token formatting enabled for ALFWorld prompts")
        else:
            logger.info("CoT token formatting disabled for ALFWorld prompts")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        backend = args.backend if args and args.backend else config.get('generator', {}).get('backend', 'api')
        self.llm_client = LLMClientFactory.create_client(config, backend)

        logger.info(
            f"Initialized {self.__class__.__name__} with backend: {backend}, "
            f"output_format: {self.output_format}, filter_success: {self.filter_success}, "
            f"task_source: {self.task_source}"
        )

    def get_output_filename(self, split: str) -> Path:
        if self.dataset_name:
            base_name = f"{self.dataset_name}_{split}"
        else:
            dataset_name = self.__class__.__name__.replace('Generator', '')
            base_name = f"{dataset_name}_{split}"
        
        base_name = base_name.lower()
        
        return self.output_dir / f"{base_name}.json"

    @abstractmethod
    def setup_environment(self, task: Dict[str, Any], split: Optional[str] = None) -> Any:
        pass

    def load_builtin_tasks(self) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            f"Task source '{self.task_source}' is not a file. "
            f"{self.__class__.__name__} must override load_builtin_tasks()."
        )

    def enrich_metadata(self, result: Dict[str, Any], episode_state: Dict[str, Any]) -> None:
        pass

    def get_expert_action(
        self,
        observation: str,
        goal: str,
        admissible_commands: List[str],
        history: List[Dict[str, str]]
    ) -> str:
        return self.get_action_with_fallback(
            observation=observation,
            goal=goal,
            admissible_commands=admissible_commands,
            history=history,
            fallback_policy=self.fallback_policy
        )

    def run_episode(self, task: Dict[str, Any], split: Optional[str] = None) -> Dict[str, Any]:
        pass

    def convert_trajectory_to_messages(self, steps: List[Dict[str, Any]], goal: str) -> List[Dict[str, str]]:
        messages = []
        for i, step in enumerate(steps):
            user_content = step["observation"]
            if i == 0:
                user_content = f"Goal: {goal}\n\nObservation: {user_content}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": step["action"]})
        return messages

    def build_react_prompt(
        self,
        observation: str,
        goal: str,
        admissible_commands: List[str],
        history: List[Dict[str, str]],
        few_shot_example: str = ""
    ) -> str:
        truncated = history[-3:] if len(history) > 3 else history
        history_str = ""
        for turn in truncated:
            history_str += f"Observation: {turn['observation']}\nAct: {turn['action']}\n\n"

        admissible_str = "\n".join(f"- {cmd}" for cmd in admissible_commands[:10])

        prompt = f"""You are an AI assistant playing a text-based game. Your goal is to complete a task.

You must reason step-by-step, then output a precise action. Format:
Think: [Your reasoning]
Act: [Your chosen action]

{few_shot_example}

Goal: {goal}

{history_str}
Current observation: {observation}

Available commands:
{admissible_str}

Think:"""
        return prompt

    def parse_action_from_response(self, response: str, admissible_commands: List[str]) -> Optional[str]:
        lines = response.split("\n")
        action = None
        for line in reversed(lines):
            if line.strip().startswith("Act:"):
                action = line.replace("Act:", "").strip()
                break
        if action is None:
            non_empty = [l.strip() for l in lines if l.strip()]
            if non_empty:
                action = non_empty[-1]
        if action and action in admissible_commands:
            return action
        return None

    def get_action_with_fallback(
        self,
        observation: str,
        goal: str,
        admissible_commands: List[str],
        history: List[Dict[str, str]],
        fallback_policy: Optional[Callable] = None
    ) -> str:
        try:
            prompt = self.build_react_prompt(observation, goal, admissible_commands, history)
            if self.show_prompt:
                logger.info(f"Prompt: {prompt}")
            response = self.llm_client.get_response(prompt)
            if self.show_response:
                logger.info(f"Response: {response}")
            action = self.parse_action_from_response(response, admissible_commands)
            if self.show_action:
                logger.info(f"Action: {action}")
            if action:
                return action
        except Exception as e:
            if "maximum context length" in str(e) or "input tokens" in str(e):
                logger.warning(f"Context length exceeded. Falling back")
            else:
                logger.warning(f"LLM expert failed: {e}")

        if fallback_policy:
            logger.info("Using fallback policy")
            try:
                action = fallback_policy(observation, goal, admissible_commands, history)
                if action and action in admissible_commands:
                    return action
            except Exception as e:
                logger.warning(f"Fallback policy failed: {e}")

        if admissible_commands:
            action = random.choice(admissible_commands)
            logger.warning(f"No valid action from LLM or fallback; using random action: {action}")
            return action

        raise RuntimeError("No admissible commands available and all experts failed")

    def load_tasks(self, num_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        if os.path.exists(self.task_source):
            with open(self.task_source, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            if not isinstance(tasks, list):
                raise ValueError("Task JSON file must contain a list of task objects")
            if num_tasks is not None and num_tasks < len(tasks):
                tasks = tasks[:num_tasks]
            for idx, task in enumerate(tasks):
                if "id" not in task:
                    task["id"] = f"{self.__class__.__name__}_task_{idx}"
            logger.info(f"Loaded {len(tasks)} tasks from JSON file: {self.task_source}")
            return tasks
        else:
            tasks = self.load_builtin_tasks()
            if num_tasks is not None and num_tasks < len(tasks):
                tasks = tasks[:num_tasks]
            return tasks

    def save_trajectories(self, trajectories: List[Dict[str, Any]], output_filename: str) -> None:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(trajectories)} trajectories to {output_filename}")

    def generate(self, split: str, num_examples: Optional[int] = None) -> List[Dict]:
        output_filename = self.get_output_filename(split)
        tasks = self.load_tasks(num_tasks=num_examples)
        logger.info(f"Generating trajectories for {len(tasks)} tasks")

        all_results = []
        successful_episodes = 0
        for idx, task in enumerate(tasks):
            task_id = task.get("id", f"task_{idx}")
            goal = task.get("goal", "No goal")
            logger.info(f"Task {idx+1}/{len(tasks)} (ID: {task_id}): {goal[:60]}...")
            try:
                result = self.run_episode(task, split=split)
                result.setdefault("task_id", task_id)
                result.setdefault("goal", goal)
                all_results.append(result)
                if result["success"]:
                    successful_episodes += 1
                    logger.info(f"  Success in {result['total_steps']} steps")
                else:
                    logger.warning(f"  Failed after {result['total_steps']} steps")
            except Exception as e:
                logger.error(f"  Error: {e}", exc_info=True)
                all_results.append({
                    "task_id": task_id, "goal": goal,
                    "success": False, "total_steps": 0,
                    "error": str(e)
                })

        if self.filter_success:
            trajectories = [r for r in all_results if r.get("success", False)]
            logger.info(f"Filtered trajectories: kept {len(trajectories)}/{len(all_results)} successful episodes")
        else:
            trajectories = all_results

        self.save_trajectories(trajectories, output_filename)
        logger.info(f"Generation complete. Success rate: {successful_episodes}/{len(tasks)} ({successful_episodes/len(tasks)*100:.1f}%)")

def main(args):
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.warning(f"Config file not found: {e}")
        config = {'generator': {'api': {}, 'vllm': {}}}
    
    if args.debug is not None:
        config['debug'] = args.debug
    
    generator_config = config.setdefault('generator', {})

    if args.backend is not None:
        backend = args.backend
        generator_config['backend'] = backend
        logger.info(f"Using backend from command line: {backend}")
    elif 'backend' in generator_config:
        backend = generator_config['backend']
        logger.info(f"Using backend from config: {backend}")
    else:
        backend = "api"
        generator_config['backend'] = backend
        logger.info(f"Using default backend: {backend}")

    if backend == 'api':
        llm_config = generator_config.setdefault('api', {})
    else:
        llm_config = generator_config.setdefault('vllm', {})
    
    batch_size = config['common'].get('batch_size', 64) if not args.batch_size else args.batch_size
    generator_config['batch_size'] = batch_size
    llm_config['batch_size'] = batch_size
    llm_config['temperature'] = generator_config['temperature']
    llm_config['max_retries'] = generator_config['max_retries']
    
    if backend == 'api':
        if args.api_key:
            llm_config['api_key'] = args.api_key
        if args.api_base:
            llm_config['api_base'] = args.api_base
        if args.model:
            llm_config['model'] = args.model
        if 'model' not in llm_config or not llm_config['model']:
            llm_config['model'] = LLM_API_MODEL
    elif backend == 'vllm':
        if args.model:
            llm_config['model'] = args.model
        elif 'model' not in llm_config or not llm_config['model']:
            logger.error("vLLM backend requires a model path. Use --model to specify a HuggingFace model path.")
            return
    else:
        logger.error(f"Unknown backend: {backend}. Must be 'api' or 'vllm'.")
        return

    if args.temperature is not None:
        generator_config['temperature'] = args.temperature
    elif 'temperature' not in generator_config:
        generator_config['temperature'] = 0.3

    if args.output_dir:
        generator_config['output_dir'] = args.output_dir
    
    if args.max_retries:
        generator_config['max_retries'] = args.max_retries
    elif 'max_retries' not in generator_config:
        generator_config['max_retries'] = 20
    
    if args.validate is not None:
        generator_config['validate_cot_steps'] = args.validate
    
    if args.dry_run:
        config['dry_run'] = True
        logger.info("Dry run completed - configuration is valid")
        return

    datasets_config = load_datasets_config()
    if args.dataset not in datasets_config:
        logger.error(f"Dataset '{args.dataset}' not found in datasets.yaml")
        return
    dataset_config = datasets_config[args.dataset]

    from dataset import GENERATOR_MAP
    if args.dataset not in GENERATOR_MAP:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available: {list(GENERATOR_MAP.keys())}")

    is_interactive = dataset_config.get('interactive', False)

    logger.info(f"{'='*60}")
    logger.info(f"DATA GENERATION")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Backend: {backend}")
    logger.info(f"Split: {args.mode}")
    if backend == 'api':
        logger.info(f"API Service: {llm_config.get('api_base')}")
        logger.info(f"Model: {llm_config.get('model')}")
    else:
        logger.info(f"Model: {llm_config.get('model', 'Not specified')}")
        logger.info(f"Batch Size: {llm_config.get('batch_size')}")
    logger.info(f"Interactive: {is_interactive}")
    if not is_interactive:
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Examples: {args.num_examples or 'all'}")
    else:
        logger.info(f"Max Steps: {dataset_config.get('max_steps', 50)}")
        logger.info(f"Output Format: {dataset_config.get('output_format', 'messages')}")
    logger.info(f"{'='*60}")

    if is_interactive:
        original_argv = sys.argv
        sys.argv = [original_argv[0]]
        try:
            generator = GENERATOR_MAP[args.dataset](config, dataset_config, args)
            generator.generate(args.mode, args.num_examples)
        finally:
            sys.argv = original_argv
        logger.info(f"Generation Complete for {args.dataset} ({args.mode})")
    else:
        cot_gen = CoTGenerator(config, backend)
        logger.info(f"✓ LLM service configured with backend: {backend}")
        generator = GENERATOR_MAP[args.dataset](cot_gen, config, args)
        results = generator.generate(args.mode, args.num_examples)
        if args.mode == 'train':
            cot_gen.llm_client.print_summary()
        logger.info(f"Generation Complete! Generated {len(results)} examples for {args.dataset} ({args.mode})")

if __name__ == '__main__':
    parser = HfArgumentParser(GeneratorArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
