# Task
Below are some claims from the responsdent with the following demographic feature:
- {{ feature_category }}: {{ feature_label }}

Please provide a concise summary that captures the key perspectives expressed in the claims.

# Claims
{% for claim in claims %}   
- **Claim**: {{ claim }}  
{% endfor %}

# Output Format (Required)
- Summary: Provide a 2-3 sentence synthesis of the respondent's views, clearly identifying key themes, contradictions, or tensions. Explicitly state how many claims support one perspective versus how many support an opposing or contrasting view, if applicable.