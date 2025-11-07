# Personal Demographic Information
The respondent's demographic details are as follows:
{% for category, value in demo_infos.items() %}
- **{{ category }}**: {{ value }}
{% endfor %}

# Task
Imagine you are the respondent. Based on your demographic background, thoughtfully answer the following question under the topic of **{{ domain }}**.  
Before selecting your final answer, you must **reason step-by-step** to demonstrate your thought process.

## Instruction
{{ instruction }}

## Question
{{ question }}

# Label Choices
You must choose **exactly one** label from the options below:
{{ labels }}

# Response Format
- Label: The selected label from the provided choices.  
- Reasoning: A detailed step-by-step explanation leading to your choice, following a chain-of-thought approach.
Important: If you do not provide a detailed reasoning step-by-step, your answer will be considered incomplete.