---
description: "Generate one or more technical blog articles from code. Usage: /code-to-blog <path> [--topics t1,t2,...]"
argument-hint: "Path to file or folder. Optional: --topics config,validation,prefect"
agent: "agent"
---

Generate publication-ready technical blog articles for: $input

Follow the instructions in [the blog skill](../skills/blog/SKILL.md).

Important execution rules:
- Do not print full articles in chat.
- If the user provides topics via `--topics`, use them exactly.
- If topics are not provided, infer 3 to 6 distinct blog-worthy topics from the code.
- Create one markdown file per topic at:
	- blogs/<topic-slug>-blog.md
- Also create or overwrite a manifest file at:
	- blogs/generated-blogs-index.md
- The manifest must include:
	- the source path analyzed
	- the topic list
	- generated file paths
	- a one-line summary for each topic
- In chat, return only:
	- the total number of posts generated
	- the list of saved file paths
	- a 1-2 line overall summary