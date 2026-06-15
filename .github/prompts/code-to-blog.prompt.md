---
description: "Generate a technical blog article from code. Usage: /code-to-blog <path>"
argument-hint: "Path to a file or folder to analyze (e.g. stockmonitorapp/models/moving_average.py)"
agent: "agent"
---

Generate a publication-ready technical blog article for: $input

Follow the instructions in [the blog skill](../skills/blog/SKILL.md).

Important execution rules:
- Do not print the full article in chat.
- Create or overwrite a markdown file in the workspace at:
	- blogs/<topic>-blog.md
- Put the complete article content in that file.
- In chat, return only:
	- the saved file path
	- a 1-2 line summary of what was generated