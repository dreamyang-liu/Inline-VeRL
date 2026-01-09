# Copyright 2024 Mengyang Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import json
import re

from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.reward_score.sandbox_fusion.utils import call_sandbox_api

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


SCRATCH_START_TAG = "<scratch>"
SCRATCH_END_TAG = "</scratch>"

SCRATCH_RESULT_START_TAG = "<scratch_result>"
SCRATCH_RESULT_END_TAG = "</scratch_result>"

SANDBOX_FUSION_URL = ""


def extract_code(text: str):
    pattern = r'```(?:python)?\n?(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text


def extract_scratch_block(text: str) -> str:
    pattern = rf'{re.escape(SCRATCH_START_TAG)}(.*?)(?:{re.escape(SCRATCH_END_TAG)}|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return None

def run_scratch(text: str):
    scratch_block = extract_scratch_block(text)
    code = extract_code(scratch_block)
    resp, err = call_sandbox_api(
        SANDBOX_FUSION_URL,
        code,
        run_timeout=10,
        memory_limit_mb=100
    )
    if resp is None:
        return f"{SCRATCH_RESULT_START_TAG}{str(err)}{SCRATCH_RESULT_END_TAG}"
    else:
        return f"{SCRATCH_RESULT_START_TAG}{json.dumps(resp)}{SCRATCH_RESULT_END_TAG}"


@register("inline_agent")
class InlineAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        messages.insert(0, {
            "role": "system",
            "content": f"You are a math assistant to help resolve math questions. IMPORTANT! You MUSE FOLLOW THE FOLLOWING RULES: Before doing any action on some critical steps to solve the problem, you MUST use the {SCRATCH_START_TAG}...{SCRATCH_END_TAG} to write easy thinking or calculation to verify your draft thoughts. You need to frequently use it to make sure you can generate verifiable steps and results."
        })

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        response_masks = []
        sampling_params["stop"] = [SCRATCH_END_TAG]
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )

            response_masks.extend([1] * len(output.token_ids))

            while self.tokenizer.decode(output.token_ids).rstrip().endswith(SCRATCH_END_TAG):
                output_str = self.tokenizer.decode(output.token_ids)
                # scratch_resp = run_scratch(output_str)
                scratch_resp = f"{SCRATCH_RESULT_START_TAG}Correct{SCRATCH_RESULT_END_TAG}"
                scratch_resp_token_id = self.tokenizer.encode(scratch_resp, add_special_tokens=False)
                extended_prompt = prompt_ids + output.token_ids + scratch_resp_token_id
                response_masks.extend([0] * len(scratch_resp_token_id))

                resume_output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=extended_prompt,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                )
                response_masks.extend([1] * len(resume_output.token_ids))
                # Combine the outputs
                output.token_ids = output.token_ids + scratch_resp_token_id + resume_output.token_ids
                if output.log_probs and resume_output.log_probs:
                    output.log_probs = output.log_probs + [None] * len(scratch_resp_token_id) + resume_output.log_probs
                if output.routed_experts and resume_output.routed_experts: 
                    output.routed_experts = output.routed_experts + [None] * len(scratch_resp_token_id) + resume_output.routed_experts
                # print(self.tokenizer.decode(output.token_ids))
                # raise RuntimeError("Debug")
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_masks[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        return output