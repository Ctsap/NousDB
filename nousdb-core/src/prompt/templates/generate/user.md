# User Prompt: Semantic Generate Task

## Task Description

You are generating new rows for a table. Each row is a new record that you compose from your own knowledge so that it satisfies the generation task described below.

1. **Count**: Return exactly the number of objects requested in the last section
2. **Field names**: Use exactly the field names listed under Output Fields
3. **Types**: Match each field's type — `string` values are quoted, `integer` and `number` values are bare numerals, `boolean` values are `true` or `false`
4. **Distinct**: Every row must differ from every other row; do not repeat a record or make near-copies of one
5. **Faithful**: Every row must satisfy the generation task
6. **Pure JSON**: No additional text, explanations, or formatting outside the JSON array

## Generation Task

**Generate**: {{instruction}}

## Output Fields

{% for field in fields %}- `{{ field.name }}` ({{ field.type }})
{% endfor %}
## Required Output Format

Respond with a JSON array of objects, one per generated row. For two rows, a well-formed response looks like this:

```json
{{ format_example }}
```
{% if examples %}
## Example Rows (illustrative — match their style and range, do not copy them)

{% for example in examples %}{{ example }}
{% endfor %}{% endif %}
## Rows to Generate

Imagine the complete table of **{{ total_rows }}** rows listed in order. Return rows **{{ row_start }}–{{ row_end }}** of that table: exactly **{{ row_count }}** objects.
