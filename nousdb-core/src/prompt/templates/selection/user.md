# User Prompt: Semantic Selection Task

## Task Description

You are evaluating rows of data to determine which ones match a specific filtering condition. For each row, you must decide whether it satisfies the condition and return a boolean value.

1. **Count**: Return exactly one object per input row
2. **Type**: Use JSON boolean literals (`true` or `false`), not strings like `"true"` or `"false"`
3. {% if reasoning_mode == "rationale" %}**Field names**: Each object holds `rationale`, a short string, first, then `matches_filter`{% else %}**Field name**: Always use `matches_filter` as the field name{% endif %}
4. **Order**: First object corresponds to Row 1, second to Row 2, etc.
5. **Pure JSON**: No additional text, explanations, or formatting outside the JSON array
6. **Consistency**: Apply the same evaluation criteria to all rows

## Filtering Condition

**Evaluate**: {{expression}}

This is the semantic condition you must evaluate for each row. A row should return true if it matches this condition, and false otherwise.

## Required Output Format

You must respond with a JSON array containing one object per input row. Each object must have this exact structure:

```json
{% if reasoning_mode == "rationale" %}{"rationale": "One short sentence with what decided this row.", "matches_filter": true}{% else %}{"matches_filter": true}{% endif %}
```

or

```json
{% if reasoning_mode == "rationale" %}{"rationale": "One short sentence with what decided this row.", "matches_filter": false}{% else %}{"matches_filter": false}{% endif %}
```

## Data to Evaluate

You will evaluate **{{row_count}}** rows. You must return exactly **{{row_count}}** boolean decisions, one for each row in the order presented.

{% for item in batch_data %}
{{ loop.index }}. {% for key in item %}{{ key }}: {{ item[key] }}{% if not loop.last %}, {% endif %}{% endfor %}
{% endfor %}
