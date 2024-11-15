---
title: "LLM for Conventional Commit Messages — WIP"
date: "2024-11-15"
summary: "Using LLMs to generate conventional commit messages"
description: "An exploration of how LLMs can be used to generate conventional commit messages"
toc: false
readTime: true
autonumber: false
math: true
tags: ["Structured Generation", "LLM"]
showTags: false
hideBackToTop: false
---

# LLM for Conventional Commit Messages

In software development, writing clear and concise commit messages is crucial for maintaining a well-documented project history. 
As a Data Scientist, I am not used to write **good** commit messages.

Conventional commit messages follow a standardized format that helps in understanding the nature of changes, automating release notes, and improving collaboration among team members. 
However, consistently crafting these messages can be $ \textcolor{black}{\raisebox{.5pt}{\textcircled{\scriptsize 1}}} $ time-consuming and $ \textcolor{black}{\raisebox{.5pt}{\textcircled{\scriptsize 2}}} $ prone to human error.

Leveraging Large Language Models (LLMs) for generating conventional commit messages can significantly streamline this process. LLMs, trained on vast amounts of text data, can understand the context of code changes and generate appropriate commit messages that adhere to the conventional format. This not only saves time but also ensures consistency and accuracy in commit messages, making the project history more readable and maintainable.

So, let's see how we can use LLMs to generate conventional commit messages. 

Our recipe will be the following:

0. Leverage `git diff` to get the changes in the commit.
1. Generate a prompt to instruct the LLM.
2. Structure the output to be in the conventional commit message format.
4. Use the LLM to generate conventional commit messages

This post is inspired by [this gist](https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285) by [@karpathy](https://github.com/karpathy).

---
### Conventional Commit Format

The conventional commit format consists of a type, a scope, and a subject, in that order.

```bash
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Document can be found [here](https://www.conventionalcommits.org/en/v1.0.0/).

Let's see some examples from the documentation: 

```
feat(lang): add Polish language
```
```
fix: prevent racing of requests

Introduce a request id and a reference to latest request. Dismiss
incoming responses other than from latest request.

Remove timeouts which were used to mitigate the racing issue but are
obsolete now.

Reviewed-by: Z
Refs: #123
```


By the way the documentation is well written, contains a lot of examples and is easy to understand. 

It may be reused in the prompt to guide the LLM to generate conventional commit messages.

---
### Structured Generation for LLMs

Structured generation for Large Language Models (LLMs) refers to the process of generating text that adheres to a specific format or structure. This is particularly useful when you want the output to follow a predefined template, such as conventional commit messages.

There are two main approaches to structured generation that I am aware of:

- llama.cpp's grammar-based sampling APIs (TODO)
- Outlines Approaches (TODO)
- More? (TODO)

The common implementation is to use the `json` format or pydantic to structure the output. 

For example with [instructor](https://github.com/instructor-ai/instructor) we can do the following:

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI


# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int


# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)

print(user_info.name)
#> John Doe
print(user_info.age)
#> 30
```

Note that there is also the library called [outlines](https://github.com/dottxt-ai/outlines) that is doing something similar. 
But during my experimentation, I struggle a bit with outlines with local models served via [ollama](https://github.com/ollama/ollama) or [lmstudio](https://lmstudio.ai). 


So, let's start by building the schema of Conventional Commit Message.

```python
from pydantic import BaseModel, Field
from typing import Optional

class ConventionalCommit(BaseModel):
    type: str = Field(
        ...,
        description="The type of change (e.g., feat, fix, chore, refactor).",
        regex="^(feat|fix|chore|refactor|test|docs|style|perf|ci|build|revert)$",
    )
    scope: Optional[str] = Field(
        None,
        description="The scope of the change (e.g., a specific module or component)."
    )
    description: str = Field(
        ...,
        description="A short description of the change.",
        max_length=72
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

# Example usage
example_commit = ConventionalCommit(
    type="feat",
    scope="auth",
    description="Add OAuth2 login support",
    body="This change integrates OAuth2 login with Google and Facebook.",
    footer="BREAKING CHANGE: Updated login API; old tokens are invalidated."
)

print(example_commit.formatted)
```

---
### Prompt Engineering

**First Try.**

```markdown
Generate a Conventional Commit message following this structure:

1. **Type**: The type of change (choose one of: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`, `build`, `revert`).
2. **Scope**: The specific module or component affected by the change (optional).
3. **Description**: A concise summary of the change in **present tense**, limited to 72 characters.
4. **Body**: A detailed explanation of **what changed** and **why** (optional, multi-line allowed).
5. **Footer**: Additional information like breaking changes or issue references (optional, e.g., "BREAKING CHANGE: ...", "Fixes #123").

**Examples**:
1. `feat(auth): add OAuth2 login support`
   - Body: This integrates OAuth2 login for Google and Facebook.
   - Footer: BREAKING CHANGE: Updated login API; old tokens are invalidated.

2. `fix(ui): resolve button alignment issue on mobile`
   - Body: Adjusted CSS to fix button alignment across all mobile devices.
   - Footer: Fixes #456.

**Task**: Using the structure above, write a commit message for the following git diff:

{{ GIT_DIFF_OUTPUT }}

Make sure the commit is concise, clear, and follows the **Conventional Commit** guidelines.
```

---
### Python Application

[...]