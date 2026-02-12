SYSTEM_PROMPT = """\
Imagine you are a web navigation assistant performing a task step by step. Your current role is to determine whether clarification is needed before taking the next action.
"""

USER_PROMPT_PROACT_SCREENSHOT = """\
Task: {task}

Previous Actions:
{previous_actions}

Previous Conversations:
{previous_conversations}

The screenshot below shows the webpage you see.

Based on the task, previous actions, previous conversations, and the current webpage, determine if clarification is needed.

If clarification is needed, please answer "Yes" and ask a clarification question. If no clarification is needed, answer "No".
"""

USER_PROMPT_PROACT_HTML = """\
Task: {task}

Previous Actions:
{previous_actions}

Previous Conversations:
{previous_conversations}

The HTML content of the webpage:
{html_content}

Based on the task, previous actions, previous conversations, and the current webpage, determine if clarification is needed.

If clarification is needed, please answer "Yes" and ask a clarification question. If no clarification is needed, answer "No".
"""

USER_PROMPT_PROACT_BOTH = """\
Task: {task}

Previous Actions:
{previous_actions}

Previous Conversations:
{previous_conversations}

The screenshot below shows the webpage you see.

The HTML content of the webpage:
{html_content}

Based on the task, previous actions, previous conversations, the current webpage, determine if clarification is needed.

If clarification is needed, please answer "Yes" and ask a clarification question. If no clarification is needed, answer "No".
"""

USER_PROMPT_CONV_ONLY = """\
Task: {task}

Previous Conversations:
{previous_conversations}

Based on the task and previous conversations, determine if clarification is needed.

If clarification is needed, please answer "Yes" and ask a clarification question. If no clarification is needed, answer "No".
"""

USER_PROMPT_ENV_ONLY_SCREENSHOT = """\
Task: {task}

Previous Actions:
{previous_actions}

The screenshot below shows the webpage you see.

Based on the task, previous actions, and the current webpage, determine if clarification is needed.

If clarification is needed, please answer "Yes" and ask a clarification question. If no clarification is needed, answer "No".
"""

USER_PROMPT_ENV_ONLY_HTML = """\
Task: {task}

Previous Actions:
{previous_actions}

The HTML content of the webpage:
{html_content}

Based on the task, previous actions, and the current webpage, determine if clarification is needed.

If clarification is needed, please answer "Yes" and ask a clarification question. If no clarification is needed, answer "No".
"""

USER_PROMPT_ENV_ONLY_BOTH = """\
Task: {task}

Previous Actions:
{previous_actions}

The screenshot below shows the webpage you see.

The HTML content of the webpage:
{html_content}

Based on the task, previous actions, and the current webpage, determine if clarification is needed.

If clarification is needed, please answer "Yes" and ask a clarification question. If no clarification is needed, answer "No".
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


def get_finetuning_prompt(mode, env, task, previous_conversations=None, previous_actions=None, html_content=None):

    if mode == "proact":
        if env == "html":
            user_prompt_template = USER_PROMPT_PROACT_HTML
        elif env == "both":
            user_prompt_template = USER_PROMPT_PROACT_BOTH
        else:
            user_prompt_template = USER_PROMPT_PROACT_SCREENSHOT
    elif mode == "env_only":
        if env == "html":
            user_prompt_template = USER_PROMPT_ENV_ONLY_HTML
        elif env == "both":
            user_prompt_template = USER_PROMPT_ENV_ONLY_BOTH
        else:
            user_prompt_template = USER_PROMPT_ENV_ONLY_SCREENSHOT
    elif mode == "conv_only":
        user_prompt_template = USER_PROMPT_CONV_ONLY
    else:
        raise ValueError(f"Unknown mode: {mode}")

    formatted_conversations = format_conversations(previous_conversations)
    formatted_actions = format_actions(previous_actions)

    prompt_kwargs = {"task": task}

    if mode != "conv_only":
        prompt_kwargs["previous_actions"] = formatted_actions

    if mode != "env_only":
        prompt_kwargs["previous_conversations"] = formatted_conversations

    if env in ["html", "both"] and html_content:
        prompt_kwargs["html_content"] = html_content

    user_prompt = user_prompt_template.format(**prompt_kwargs)

    return SYSTEM_PROMPT, user_prompt