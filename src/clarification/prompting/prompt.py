"""
============================================
System Prompts
============================================
"""

SYSTEM_PROMPT_BASE = """\
Imagine you are a web navigation assistant performing a task step by step. Your current role is to determine whether clarification is needed before taking the next action.

{access_info} Based on all available information, decide if the next action can be taken or if you need to ask the user for clarification due to missing, ambiguous, or infeasible details.
"""

PROACT_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.format(
    access_info="You have access to the task description, previous actions, previous conversations, and the current webpage through screenshot."
)

PROACT_HTML_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.format(
    access_info="You have access to the task description, previous actions, previous conversations, and the current webpage through HTML."
)

PROACT_BOTH_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.format(
    access_info="You have access to the task description, previous actions, previous conversations, and the current webpage through both screenshot and HTML."
)

ENVIRONMENT_ONLY_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.format(
    access_info="You have access to the task description, previous actions, and the current webpage through screenshot."
)

ENVIRONMENT_ONLY_HTML_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.format(
    access_info="You have access to the task description, previous actions, and the current webpage through HTML."
)

ENVIRONMENT_ONLY_BOTH_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.format(
    access_info="You have access to the task description, previous actions, and the current webpage through both screenshot and HTML."
)

CONV_ONLY_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.format(
    access_info="You have access to the task description and previous conversations."
)

"""
============================================
User Prompts
============================================
"""

PROACT_USER_PROMPT = """\
You need to analyze whether the following task requires clarification: {task}

Previous Actions: {previous_actions}

Previous Conversations: {previous_conversations}

The screenshot below shows the webpage you see.

Follow this guidance to think step by step:

(Task Analysis)
Examine the task description to understand what the user wants to accomplish.

(Webpage Analysis)
Analyze the screenshot to understand what the webpage is, what elements are available, and how they relate to the task.

(Previous Actions Analysis)
Review the actions already taken to understand the progress toward task completion and whether the navigation path aligns with the task.

(Previous Conversations Analysis)
Check previous conversations to see what clarifications have been made and what specific information the user has already provided.

(Decision)
Based on analysis above, determine if you can proceed with the next action or need to ask for clarification. If clarification is needed, first check whether previous conversations have already addressed this issue. Only ask for clarification if the issue is genuinely unresolved.

Please strictly respond in following JSON format:
{{
    "clarification_need": true/false,
    "clarification_question": "specific question to ask if clarification is needed, otherwise null",
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}
"""

PROACT_HTML_USER_PROMPT = """\
You need to analyze whether the following task requires clarification: {task}

Previous Actions: {previous_actions}

Previous Conversations: {previous_conversations}

The HTML content of the webpage:
{html_content}

Follow this guidance to think step by step:

(Task Analysis)
Examine the task description to understand what the user wants to accomplish.

(Webpage Analysis)
Analyze the HTML to understand what the webpage is, what elements are available, and how they relate to the task.

(Previous Actions Analysis)
Review the actions already taken to understand the progress toward task completion and whether the navigation path aligns with the task.

(Previous Conversations Analysis)
Check previous conversations to see what clarifications have been made and what specific information the user has already provided.

(Decision)
Based on analysis above, determine if you can proceed with the next action or need to ask for clarification. If clarification is needed, first check whether previous conversations have already addressed this issue. Only ask for clarification if the issue is genuinely unresolved.

Please strictly respond in following JSON format:
{{
    "clarification_need": true/false,
    "clarification_question": "specific question to ask if clarification is needed, otherwise null",
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}
"""

PROACT_BOTH_USER_PROMPT = """\
You need to analyze whether the following task requires clarification: {task}

Previous Actions: {previous_actions}

Previous Conversations: {previous_conversations}

The screenshot below shows the webpage you see.

The HTML content of the webpage:
{html_content}

Follow this guidance to think step by step:

(Task Analysis)
Examine the task description to understand what the user wants to accomplish.

(Webpage Analysis)
Analyze both the screenshot and HTML to understand what the webpage is, what elements are available, and how they relate to the task.

(Previous Actions Analysis)
Review the actions already taken to understand the progress toward task completion and whether the navigation path aligns with the task.

(Previous Conversations Analysis)
Check previous conversations to see what clarifications have been made and what specific information the user has already provided.

(Decision)
Based on analysis above, determine if you can proceed with the next action or need to ask for clarification. If clarification is needed, first check whether previous conversations have already addressed this issue. Only ask for clarification if the issue is genuinely unresolved.

Please strictly respond in following JSON format:
{{
    "clarification_need": true/false,
    "clarification_question": "specific question to ask if clarification is needed, otherwise null",
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}
"""

ENVIRONMENT_ONLY_USER_PROMPT = """\
You need to analyze whether the following task requires clarification: {task}

Previous Actions: {previous_actions}

The screenshot below shows the webpage you see.

Follow this guidance to think step by step:

(Task Analysis)
Examine the task description to understand what the user wants to accomplish.

(Webpage Analysis)
Analyze the screenshot to understand what the webpage is, what elements are available, and how they relate to the task.

(Previous Actions Analysis)
Review the actions already taken to understand the progress toward task completion and whether the navigation path aligns with the task.

(Decision)
Based on analysis above, determine if you can proceed with the next action or need to ask for clarification. Only ask for clarification if the issue is genuinely unresolved.

Please strictly respond in following JSON format:
{{
    "clarification_need": true/false,
    "clarification_question": "specific question to ask if clarification is needed, otherwise null",
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}
"""

ENVIRONMENT_ONLY_HTML_USER_PROMPT = """\
You need to analyze whether the following task requires clarification: {task}

Previous Actions: {previous_actions}

The HTML content of the webpage:
{html_content}

Follow this guidance to think step by step:

(Task Analysis)
Examine the task description to understand what the user wants to accomplish.

(Webpage Analysis)
Analyze the HTML to understand what the webpage is, what elements are available, and how they relate to the task.

(Previous Actions Analysis)
Review the actions already taken to understand the progress toward task completion and whether the navigation path aligns with the task.

(Decision)
Based on analysis above, determine if you can proceed with the next action or need to ask for clarification. Only ask for clarification if the issue is genuinely unresolved.

Please strictly respond in following JSON format:
{{
    "clarification_need": true/false,
    "clarification_question": "specific question to ask if clarification is needed, otherwise null",
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}
"""

ENVIRONMENT_ONLY_BOTH_USER_PROMPT = """\
You need to analyze whether the following task requires clarification: {task}

Previous Actions: {previous_actions}

The screenshot below shows the webpage you see.

The HTML content of the webpage:
{html_content}

Follow this guidance to think step by step:

(Task Analysis)
Examine the task description to understand what the user wants to accomplish.

(Webpage Analysis)
Analyze both the screenshot and HTML to understand what the webpage is, what elements are available, and how they relate to the task.

(Previous Actions Analysis)
Review the actions already taken to understand the progress toward task completion and whether the navigation path aligns with the task.

(Decision)
Based on analysis above, determine if you can proceed with the next action or need to ask for clarification. Only ask for clarification if the issue is genuinely unresolved.

Please strictly respond in following JSON format:
{{
    "clarification_need": true/false,
    "clarification_question": "specific question to ask if clarification is needed, otherwise null",
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}
"""

CONV_ONLY_USER_PROMPT = """\
You need to analyze whether the following task requires clarification: {task}

Previous Conversations: {previous_conversations}

Follow this guidance to think step by step:

(Task Analysis)
Examine the task description to understand what the user wants to accomplish.

(Previous Conversations Analysis)
Check previous conversations to see what clarifications have been made and what specific information the user has already provided.

(Decision)
Based on analysis above, determine if you can proceed with the next action or need to ask for clarification. If clarification is needed, first check whether previous conversations have already addressed this issue. Only ask for clarification if the issue is genuinely unresolved.

Please strictly respond in following JSON format:
{{
    "clarification_need": true/false,
    "clarification_question": "specific question to ask if clarification is needed, otherwise null",
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}
"""


def format_conversations(conversations):
    if not conversations:
        return "None"

    formatted = []
    for turn in conversations:
        if 'question' in turn and 'response' in turn:
            formatted.append(f"assistant: {turn['question']}")
            formatted.append(f"user: {turn['response']}")

    return "\n".join(formatted) if formatted else "None"


def format_actions(actions):
    if not actions:
        return "None"

    return "\n".join(actions) if actions else "None"


def get_clarification_prompt(mode, env_type, task, previous_conversations, previous_actions, html_content):

    if mode == "proact":
        if env_type == "html":
            system_prompt = PROACT_HTML_SYSTEM_PROMPT
            user_prompt_template = PROACT_HTML_USER_PROMPT
        elif env_type == "both":
            system_prompt = PROACT_BOTH_SYSTEM_PROMPT
            user_prompt_template = PROACT_BOTH_USER_PROMPT
        else:
            system_prompt = PROACT_SYSTEM_PROMPT
            user_prompt_template = PROACT_USER_PROMPT
    elif mode == "env_only":
        if env_type == "html":
            system_prompt = ENVIRONMENT_ONLY_HTML_SYSTEM_PROMPT
            user_prompt_template = ENVIRONMENT_ONLY_HTML_USER_PROMPT
        elif env_type == "both":
            system_prompt = ENVIRONMENT_ONLY_BOTH_SYSTEM_PROMPT
            user_prompt_template = ENVIRONMENT_ONLY_BOTH_USER_PROMPT
        else:
            system_prompt = ENVIRONMENT_ONLY_SYSTEM_PROMPT
            user_prompt_template = ENVIRONMENT_ONLY_USER_PROMPT
    elif mode == "conv_only":
        system_prompt = CONV_ONLY_SYSTEM_PROMPT
        user_prompt_template = CONV_ONLY_USER_PROMPT
    else:
        raise ValueError(f"Unknown mode: {mode}")

    formatted_conversations = format_conversations(previous_conversations)
    formatted_actions = format_actions(previous_actions)

    prompt_kwargs = {"task": task}

    if mode != "conv_only":
        prompt_kwargs["previous_actions"] = formatted_actions

    if mode != "env_only":
        prompt_kwargs["previous_conversations"] = formatted_conversations

    if env_type and env_type in ["html", "both"]:
        prompt_kwargs["html_content"] = html_content

    user_prompt = user_prompt_template.format(**prompt_kwargs)

    return system_prompt, user_prompt