---
display_title: Code to Blog Generator
description: Analyzes a code file/repository and generates an engaging, well-structured technical blog post.
commands:
- /code-to-blog
---

# Instructions

When a user provides the `/code-to-blog` command and a path to a code file, follow these steps in order:

1. **Analyze the Code**: Review the provided code to understand its purpose, functionality, and the problem it solves. 
2. **Draft the Structure**: Generate an outline featuring:
   - An engaging H1 (Title)
   - A brief introduction on the problem the code solves
   - A "How it Works" section with clearly formatted markdown code blocks
   - A summary or conclusion with key takeaways
3. **Check Quality**: Ensure all included code snippets are accurate and explained in clear, developer-friendly prose. 
4. **Formatting**: Output the final article in clean Markdown, including H2/H3 tags and callout boxes for "Pro Tips."
5. **Add Visuals**: Include a placeholder for a featured image or diagram that visually represents the code's functionality or architecture.
6. **Target Platform**: Keep the content suitable for platforms like Medium and Substack, ensuring it is engaging and informative for a technical audience.

# Output Rules
1. Write in a tone that is authoritative yet accessible (like a senior developer).
2. Do not just dump the code; explain the *why* and the *how*.
3. Add a placeholder for a featured image or diagram (`[Image Placeholder]`).
