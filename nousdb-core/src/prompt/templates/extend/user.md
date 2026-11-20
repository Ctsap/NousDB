# User Prompt: Semantic Extend Task

## Task Description

You are computing new attribute values for rows of data. For each row, you must generate the requested new values based on the input data according to the semantic computation described below.

1. **Count**: Return exactly one object per input row
2. **Field names**: Use exactly the field names specified in the output schema
3. **Types**: Match the data types specified in the output schema (string, number, boolean, etc.)
4. **Order**: First object corresponds to Row 1, second to Row 2, etc.
5. **Pure JSON**: No additional text, explanations, markdown formatting, or commentary outside the JSON array
6. **Consistency**: Apply the same computation methodology to all rows
7. **Completeness**: Process every row—do not skip any rows

## Computation Task

**Compute**: {{instruction}}

## Required Output Format

You must respond with a JSON array containing one object per input row. Each object must contain the new field(s) being computed with the exact field names and types specified below.

### Output Schema

Each result object must match this schema:

{{ output_schema }}

## Data to Process

You will process **{{row_count}}** rows. You must return exactly **{{row_count}}** result objects, one for each row in the order presented.

{% for item in batch_data %}
[R.{{ loop.index }}] {% for key in item %}{{ key }}: {{ item[key] }}{% if not loop.last %}, {% endif %}{% endfor %}
{% endfor %}
