# Personal Demographic Information
The respondent's demographic details are as follows:
{% for category, value in demo_infos.items() %}
- **{{ category }}**: {{ value }}
{% endfor %}

# Task
Imagine you are the respondent. Based on your demographic background, answer the following question under the topic of **{{ domain }}**.  

## Instruction
{{ instruction }}

## Question
{{ question }}

# Label Choices
You must choose **exactly one** label from the options below:
{{ labels }}

# Response Format
- Label: The selected label from the provided choices.  
- Reasoning: An explanation leading to your choice.