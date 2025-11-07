# Demographic Profile
Below is a simulated demographic profile. Please respond as if you belong to this background:
{% for category, value in demo_infos.items() %}
- **{{ category }}**: {{ value }}
{% endfor %}

# Context
You will evaluate claims and counterpoints that reflect opinions or beliefs people might hold in the field of **{{ domain }}**. Each pair is designed to capture a possible tension or debate that may arise based on demographic perspectives.

These statements are *hypothetical and intentionally diverse* to explore how views might vary across backgrounds. Your task is not to judge them by factual accuracy, but to engage thoughtfully based on your assigned profile.

## Statements
For each of the following, consider both the claim and the counterpoint:
{% for claim, counterpoint in claims.items() %}
- **Claim**: {{ claim }}  
  - **Counterpoint**: "{{ counterpoint }}"
{% endfor %}

# Task
You are asked to select the position that would most closely align with the simulated demographic perspective above. This is a reasoned choice based on how someone from this profile might respond. In doing so, carefully consider how the specific claims made in the prompt may influence their reasoning. At the same time, critically reflect on potential counterpoints—how someone from this demographic might still be persuaded by alternative views. Your answer should weigh these tensions and offer a thoughtful justification.

# Instruction
{{ instruction }}

# Question
{{ question }}

# Label Choices
Choose **exactly one** of the following:
{{ labels }}

# Response Format (Required)
- **Label**: your selected label from above
- **Reasoning**: Step-by-step explanation of how the claims, counterpoints and simulated demographic background influence the choice. Be specific and avoid generic justifications.

> ⚠️ Incomplete responses without detailed reasoning will be considered invalid for this task.
