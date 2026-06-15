---
display_title: Code to Blog Generator
description: Analyzes a code file/repository and generates an engaging, well-structured technical blog post.
commands:
- /code-to-blog
---

# Instructions

When a user provides the `/code-to-blog` command and a path to a code file, follow these steps in order:

1. **Analyze the Code**: Review the provided code to understand its purpose, functionality, and the problem it solves. 
2. **Identify Topics**: Determine whether the code supports multiple distinct topics. If it does, split the content into separate topic-focused blog posts instead of forcing everything into one article.
3. **Draft the Structure**: For each blog post, generate an outline featuring:
   - An engaging H1 (Title)
   - A brief introduction on the problem the code solves
   - A "How it Works" section with clearly formatted markdown code blocks
   - A summary or conclusion with key takeaways
4. **Check Quality**: Ensure all included code snippets are accurate and explained in clear, developer-friendly prose.
5. **Formatting**: Output each final article in clean Markdown, including H2/H3 tags and callout boxes for "Pro Tips."
6. **Add Visuals**: Include a placeholder for a featured image or diagram that visually represents the code's functionality or architecture.
7. **Target Platform**: Keep the content suitable for platforms like Medium and Substack, ensuring it is engaging and informative for a technical audience.

## Branding Template

Use this consistent title style for each post:
- `Series: Specific Topic: SubTopic/Practical Outcome`

Use this fixed 5-sentence introduction template in order:
1. Explain why the topic matters in production workflows.
2. Explain what breaks or degrades without this method.
3. Explain what the post implements in Python/pandas.
4. Explain what practical capability the reader gains.
5. Explain where this post fits in the broader series.

## Multi-Topic Policy

When a file contains multiple blog-worthy concerns, generate separate posts around distinct themes rather than combining them into a single broad article.

Good topic boundaries include:
- configuration loading and feature toggles
- schema validation and data quality checks
- data normalization and per-entity extraction
- task orchestration and workflow design
- feature computation architecture
- output persistence and reproducibility

For each topic-specific post:
- Keep the article standalone and understandable on its own.
- Use only the code snippets relevant to that topic.
- Minimize overlap with other generated posts.
- Include topic-specific pitfalls, tradeoffs, and pro tips.
- Prefer technical depth over broad summary.

# Output Rules
1. Write in a tone that is authoritative yet accessible (like a senior developer).
2. Do not just dump the code; explain the *why* and the *how*.
3. Add a placeholder for a featured image or diagram (`[Image Placeholder]`).
4. If multiple posts are generated, ensure each has a unique title, angle, and conclusion.
