# Semantic Join Task

## Task Description

You are evaluating pairs of rows drawn from two relations to decide whether each pair satisfies a semantic join condition. Rows are listed once as reference tables — L1, L2, ... for the left relation, R1, R2, ... for the right — and the pair list at the end names which pairs to evaluate by ID.

1. **Count**: Return exactly one object per listed pair
2. **Type**: Use JSON boolean literals (`true` or `false`), not strings like `"true"` or `"false"`
3. {% if reasoning_mode == "rationale" %}**Field names**: Each object holds `rationale`, a short string, first, then `join_match`{% else %}**Field name**: Always use `join_match` as the field name{% endif %}
4. **Order**: The first object corresponds to Pair 1, the second to Pair 2, etc.
5. **Pure JSON**: No additional text, explanations, or formatting outside the JSON array
6. **Independence**: Evaluate each pair on its own, using only the referenced row data
7. **Repeats**: The same row ID may be referenced by several pairs — each pair still needs its own decision

## Join Condition

**Evaluate**: {{join_condition}}

A pair matches when the referenced left row and right row together satisfy this condition, and does not match otherwise.

## Required Output Format

Respond with a JSON array of objects, {% if reasoning_mode == "rationale" %}each holding a short `rationale` string followed by a `join_match` boolean{% else %}each holding a single `join_match` boolean{% endif %}. For two pairs, a well-formed response looks like this:

```json
{% if reasoning_mode == "rationale" %}[{"rationale": "One short sentence with what decided this pair.", "join_match": true}, {"rationale": "One short sentence with what decided this pair.", "join_match": false}]{% else %}[{"join_match": true}, {"join_match": false}]{% endif %}
```

Emit one such object per pair listed below, in the order listed.

## Left Rows

{% for row in left_rows %}
**L{{loop.index}}:**
{% for key in row %}  - {{ key }}: {{ row[key] }}
{% endfor %}
{% endfor %}

## Right Rows

{% for row in right_rows %}
**R{{loop.index}}:**
{% for key in row %}  - {{ key }}: {{ row[key] }}
{% endfor %}
{% endfor %}

## Pairs to Evaluate

Evaluate **{{pair_count}}** pairs. Return exactly **{{pair_count}}** boolean decisions, in the order listed.

{% for pair in pairs %}
- Pair {{loop.index}}: L{{pair.left_id}} ↔ R{{pair.right_id}}
{% endfor %}
