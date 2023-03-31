"""
shell-gpt: An interface to OpenAI's ChatGPT (GPT-3.5) API

This module provides a simple interface for OpenAI's ChatGPT API using Typer
as the command line interface. It supports different modes of output including
shell commands and code, and allows users to specify the desired OpenAI model
and length and other options of the output. Additionally, it supports executing
shell commands directly from the interface.

API Key is stored locally for easy use in future runs.
"""
import os
import click
import subprocess

from typing import Mapping, List

import typer

# Click is part of typer.
from click import MissingParameter, BadParameter
from sgpt import config, make_prompt, OpenAIClient
from sgpt.utils import (
    echo_chat_ids,
    echo_chat_messages,
    get_edited_prompt,
)


def get_completion(
    messages: List[Mapping[str, str]],
    temperature: float,
    top_p: float,
    caching: bool,
    chat: str,
):
    api_host = config.get("OPENAI_API_HOST")
    api_key = config.get("OPENAI_API_KEY")
    client = OpenAIClient(api_host, api_key)
    return client.get_completion(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=temperature,
        top_probability=top_p,
        caching=caching,
        chat_id=chat,
    )


@click.command()
@click.argument("prompt", nargs=-1)
@click.option(
    "--temperature",
    "-t",
    default=1.0,
    type=float,
    help="Randomness of generated output.",
)
@click.option(
    "--top-probability",
    "-p",
    default=1.0,
    type=float,
    help="Limits highest probable tokens (words).",
)
@click.option("--chat", "-c", type=str, help="Follow conversation with id (chat mode).")
@click.option(
    "--show-chat", "-s", type=str, help="Show all messages from provided chat id."
)
@click.option("--list-chat", "-l", is_flag=True, help="List all existing chat ids.")
@click.option("--shell", is_flag=True, help="Generate and execute shell command.")
@click.option("--code", is_flag=True, help="Provide code as output.")
@click.option(
    "--demo", "-d", is_flag=True, help="Shows the shell prompt but doesn't run."
)
@click.option("--editor", is_flag=True, help="Open $EDITOR to provide a prompt.")
@click.option("--no-cache", is_flag=True, help="Disable completion cache.")
def main(
    prompt,
    temperature,
    top_probability,
    chat,
    show_chat,
    list_chat,
    shell,
    code,
    demo,
    editor,
    no_cache,
) -> None:
    collectedPrompt = " ".join(prompt)

    if list_chat:
        echo_chat_ids()
        return
    if show_chat:
        echo_chat_messages(show_chat)
        return

    if not collectedPrompt and not editor:
        raise MissingParameter(param_hint="PROMPT", param_type="string")

    if editor:
        collectedPrompt = get_edited_prompt()

    if chat and OpenAIClient.chat_cache.exists(chat):
        chat_history = OpenAIClient.chat_cache.get_messages(chat)
        is_shell_chat = chat_history[0].endswith("###\nCommand:")
        is_code_chat = chat_history[0].endswith("###\nCode:")
        if is_shell_chat and code:
            raise BadParameter(
                f"Chat id:{chat} was initiated as shell assistant, can be used with --shell only"
            )
        if is_code_chat and shell:
            raise BadParameter(
                f"Chat id:{chat} was initiated as code assistant, can be used with --code only"
            )

        collectedPrompt = make_prompt.chat_mode(
            collectedPrompt, is_shell_chat, is_code_chat
        )
    else:
        collectedPrompt = make_prompt.initial(collectedPrompt, shell, code)

    completion = get_completion(
        messages=[{"role": "user", "content": collectedPrompt}],
        temperature=temperature,
        top_p=top_probability,
        caching=not no_cache,
        chat=chat,
    )

    full_completion = ""
    for word in completion:
        typer.secho(word, fg="magenta", bold=True, nl=False)
        full_completion += word

    typer.secho()
    if not code and shell:
        if demo == False:
            if "only PowerShell commands" in str(collectedPrompt):
                subprocess.run(["powershell", "-Command", full_completion])
            else:
                os.system(full_completion)


def entry_point() -> None:
    # Python package entry point defined in setup.py
    main()


if __name__ == "__main__":
    main()
