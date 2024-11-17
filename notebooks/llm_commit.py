import subprocess
import typer
import outlines                 # Prompt 
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
from llama_cpp import Llama
from openai import OpenAI
import instructor


app = typer.Typer()

# ================================
# Conventional Commit 
# ================================  


class ConventionalCommit(BaseModel):
    type: str = Field(
        ...,
        description="""
        The type of change (e.g., feat, fix, chore, refactor). 
        Should respect the regex ^(feat|fix|chore|refactor|test|docs|style|perf|ci|build|revert)$
        """,
        pattern="^(feat|fix|chore|refactor|test|docs|style|perf|ci|build|revert)$",
    )
    scope: Optional[str] = Field(
        None,
        description="The scope of the change (e.g., a specific module or component)."
    )
    description: str = Field(
        ...,
        description="A short description of the change. Max length is 72 characters.",
        max_length=150
    )
    body: Optional[str] = Field(
        None,
        description="Detailed explanation of the change."
    )
    footer: Optional[str] = Field(
        None,
        description="Additional information (e.g., breaking changes or issues fixed)."
    )

    @property
    def formatted(self) -> str:
        """
        Returns the conventional commit in the correct format.
        """
        parts = [f"{self.type}{f'({self.scope})' if self.scope else ''}: {self.description}"]
        if self.body:
            parts.append(f"\n\n{self.body}")
        if self.footer:
            parts.append(f"\n\n{self.footer}")
        return "".join(parts)

# ================================
# Git 
# ================================


def get_git_diff(path: str) -> str:
    """
    Returns the `git diff` output for a given path if it's a Git repository
    and has changes staged.

    :param path: The path to the directory or repository.
    :return: The git diff as a string, or an appropriate message if no diff is available.
    """
    path = Path(path)
    if not path.exists():
        return f"The path '{path}' does not exist."
    
    # Check if it's a git repository
    try:
        subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        return f"The path '{path}' is not a Git repository."

    # Check for staged changes
    staged_changes = subprocess.run(
        ["git", "-C", str(path), "diff", "--cached", "--name-only"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not staged_changes.stdout.strip():
        return "No staged changes found in the repository."

    # Get the git diff
    git_diff = subprocess.run(
        ["git", "-C", str(path), "diff", "--cached"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if git_diff.returncode != 0:
        return f"Error retrieving git diff: {git_diff.stderr.decode('utf-8')}"
    
    return git_diff.stdout.decode("utf-8")


# ================================
# LLM Prompt 
# ================================


@outlines.prompt
def prompt_generation_conventional_commit(instructions):
    """ 
    Instructions
    ---------------
    {{ instructions }}

    Reasoning Step
    ------------------
    Before generating the conventional commit, think step-by-step:
    0. Summarize the git diff in concise terms to remove any noises. Keep important details and metadata about files and directories.
    1. Identify the type of changes based on the git diff, choose one of: (feat|fix|chore|refactor|test|docs|style|perf|ci|build|revert).
    2. Determine the scope of the change (e.g., module, file, function).
    3. Add any additional context that provides clarity about the change.
    4. Verify that the commit message adheres to the Conventional Commit specification.
    5. If the commit message is not adhering to the Conventional Commit specification,
       revise the commit message accordingly.

    Generate Output
    ----------------
    Combine the information derived in the CoT reasoning step to generate a commit message
    adhering to the Conventional Commit specification.

    """

instruction = """ Generate a Conventional Commit message from a git diff snapshot following this structure:

1. **Type**: The type of change (choose one of: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`, `build`, `revert`).
2. **Scope**: The specific module or component affected by the change (optional).
3. **Description**: A concise summary of the change in **present tense**, limited to 72 characters.
4. **Body**: A detailed explanation of **what changed** and **why** (optional, multi-line allowed).
5. **Footer**: Additional information like breaking changes or issue references (optional, e.g., "BREAKING CHANGE: ...", "Fixes #123").

Make sure the commit is concise, clear, and follows the **Conventional Commit** guidelines.

"""

# ================================
# LLM 
# ================================  

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)


# ================================
# Front End 
# ================================


@app.command()
def git_diff(path: str):
    """
    Command to get the `git diff` for a given path.

    :param path: Path to the directory or repository.
    """
    result = get_git_diff(path)
    typer.echo(result)

@app.command()
def plan(path: str):
    """
    Command to plan a commit message for a given path.

    :param path: Path to the directory or repository.
    """
    result = get_git_diff(path)
    prompt = prompt_generation_conventional_commit(instruction)
    
    typer.echo(result)

    resp = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": result,
            }
        ],
        response_model=ConventionalCommit,
    )

    print(resp.model_dump_json(indent=2))
    #formatted = ConventionalCommit(**resp.model_dump()).formatted
    #print(resp.model_dump_json(indent=2))
    #print(formatted)

if __name__ == "__main__":
    app()
