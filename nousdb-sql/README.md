# nousdb-sql

SQL functions for NousDB's semantic operators: `EXTEND`, `SEM_FILTER`, `SEM_JOIN` and `GENERATE`.

A `NousDBContext` installs them by default, so ordinary SQL can call a language model:

```sql
SELECT *, EXTEND('classify sentiment', 'm', review) AS sentiment FROM reviews;
SELECT * FROM papers WHERE SEM_FILTER('is about databases', 'm', abstract) AND year > 2020;
SELECT * FROM a JOIN b ON SEM_JOIN('same product?', 'm', a.title, b.name);
SELECT * FROM GENERATE('20 fictional startups', 'm', 'name: Utf8, pitch: Utf8', 20);
```

Each call becomes the same logical node the DataFrame API builds (`ExtendNode`, `SemanticSelectionNode`, `SemanticJoinNode`, `GenerateNode`), so NousDB's optimizer rules and physical operators treat SQL and DataFrame queries identically. The crate depends on DataFusion and `nousdb-operators` only; it runs no model itself.

## Conventions

- Function names resolve in any case: `extend`, `EXTEND` and `Extend` give the same plan.
- Argument order is always instruction, model, then the rest.
- The instruction and the model are string literals. A column or an expression there is a plan error. The model id is resolved when the query is physically planned, so an unknown model fails then, not at SQL planning.
- Input columns are plain columns of the relation the function is applied to, with no repeats. They are bound to the operator by their bare name, so a name shared by two columns of the input is rejected (`rename one in a subquery`). Correlated (outer) columns are not supported.
- Settings that are not arguments (batch size, reasoning mode, request window) are session-level, set on `NousDBContextBuilder`.

## Functions

### `EXTEND` in a SELECT list

```
EXTEND(instruction, model, column [, column ...] [, 'type' | 'name: type'])
```

Adds one column computed by the model from the input columns.

- Output: one nullable column. The type is `Utf8` unless the trailing literal gives one. The column is named by the literal's `name:` part if present (`'stars: Int64'` gives `stars`), and an explicit `AS` alias wins over it (the model still sees the `name:` field). With neither, the column keeps DataFusion's default name for the call, `extend(<args>)`; the extend node's internal field is then called `result`. A name already used by an input column, or by an earlier `EXTEND` in the same SELECT, gets a `_1`, `_2`, ... suffix on the node's field.
- The trailing literal is recognised when the last argument is a string literal and at least one column comes before it. A literal naming more than one field is an error that points to the relation form.
- Where: in the SELECT list, also nested inside another expression there (`upper(EXTEND(...))`). The same call written twice in one SELECT list runs once. Several distinct calls stack in written order, the first one written innermost.
- Not allowed in WHERE, GROUP BY, ORDER BY or any other clause (compute it in a subquery and use the column).

```sql
SELECT *, EXTEND('rate 1 to 5', 'm', review, 'Int64') AS stars FROM reviews;
SELECT title, EXTEND('rate 1 to 5', 'm', review, 'stars: Int64') FROM reviews;
SELECT *, EXTEND('would they recommend it?', 'm', review, 'Boolean') AS rec FROM reviews;
```

To use the model's yes/no answer as a filter, compute it as a `Boolean` column in a subquery and filter on that, or use `SEM_FILTER`.

### `EXTEND` as a relation

```
FROM EXTEND(<table | (subquery)>, instruction, model, column [, column ...], 'name: type[, ...]')
```

Adds several flat columns in one request per batch of rows.

- The first argument is a table, view or CTE name (optionally schema-qualified) or a parenthesized subquery.
- The schema literal is required. Its names must not collide with a column of the input.
- Output: all input columns in their order, then the derived columns, all nullable. An alias after the call (`AS e`) works. The derived columns are ordinary columns, so `WHERE sentiment = 'positive'` and `ORDER BY` work on them.
- A table name is looked up through a hidden table function, `__nousdb_resolve_table`, because DataFusion loads only the tables it finds as relations in `FROM`. A catalog whose lookup needs I/O cannot be resolved this way and fails with an error that suggests the subquery form, `EXTEND((SELECT * FROM t), ...)`, which works for every catalog.

```sql
SELECT * FROM EXTEND(reviews, 'extract both', 'm', review, 'sentiment: Utf8, genre: Utf8');
SELECT * FROM EXTEND((SELECT * FROM reviews WHERE year > 2020), 'extract both', 'm', review, 'sentiment: Utf8, genre: Utf8');
```

It is a relation planner, not a table function, because DataFusion 55.2 looks table functions up case-sensitively. A table that happens to be named `extend`, used without parentheses, is left alone.

### `SEM_FILTER`

```
SEM_FILTER(instruction, model, column [, column ...])
```

Keeps the rows for which the model answers yes about the given columns.

- Where: a top-level condition AND-ed into `WHERE` (or `HAVING`, which takes the same path). Under `OR`, under `NOT`, inside another expression, in a SELECT list or in a JOIN's `ON` it is an error.
- Relational conditions written with it are applied first, below the semantic selection, so the model sees only the rows they keep. The exception is a correlated condition inside a subquery (one that reads a column of the outer query, `b.sku = a.id * 10` in `WHERE EXISTS (SELECT 1 FROM b WHERE b.sku = a.id * 10 AND SEM_FILTER(...))`): it stays above the semantic selection, because DataFusion's decorrelation cannot pull a condition up through a semantic node. Several `SEM_FILTER`s keep their written order, the first written innermost; the optimizer's selection reordering may change that order.
- A `SEM_FILTER` over columns of a join goes in `WHERE` above the join: `a JOIN b ON a.k = b.k WHERE SEM_FILTER('...', 'm', a.x, b.y)`. It then runs on the joined rows, once per joined row, not once per base row; to pay once per base row, filter the base table in a subquery: `FROM (SELECT * FROM a WHERE SEM_FILTER(...)) a JOIN b ON ...`.

```sql
SELECT * FROM papers WHERE SEM_FILTER('is about databases', 'm', abstract) AND year > 2020;
```

### `SEM_JOIN`

```
SEM_JOIN(instruction, model, column, column [, column ...])
```

Joins two relations on a condition the model judges per pair of rows.

- Columns: at least one from each side. They are split by side in written order. A name found on both sides must be qualified.
- Where: in an `INNER JOIN`'s `ON`, or in `WHERE` directly over a comma or `CROSS` join of two relations. Other join types, `USING`, `NATURAL` and equi-join keys in the same join are errors, as is a second `SEM_JOIN` in one join and `SEM_FILTER` inside `ON`.
- Relational conditions written with it (in `ON`, or in the same `WHERE`) are placed by the columns of this query they read, a subquery's references to this query's columns included:

  | Condition | Placed |
  |---|---|
  | reads columns of one side only (`a.price > 10`, `EXISTS (SELECT 1 FROM r WHERE r.k = b.sku)`) | a filter on that side's input, before the join |
  | reads a column of an enclosing query (the statement is a correlated subquery) | a filter above the join (above any `SEM_FILTER` too) |
  | volatile (`random() < 0.5`), or reads no column (`1 = 1`, an uncorrelated `EXISTS`) | a filter above the join |
  | reads both sides through a subquery (`EXISTS (... r.k = b.sku AND r.j = a.id)`) | a filter above the join |
  | reads both sides directly (`a.k = b.k`) | an error, because the model would run over the whole cross product; the message recommends `JOIN ... ON a.k = b.k WHERE SEM_FILTER(...)` |

  A filter above the join keeps the join's cost: the model still judges every pair of the (side-filtered) inputs.
- In `WHERE` over a comma join of more than two relations, `SEM_JOIN` joins the last relation listed with the join of all the ones before it; relations are not reordered. So the two relations it compares must be the two sides of the last comma or `CROSS` join: `FROM a, l, b WHERE SEM_JOIN('...', 'm', a.x, b.y) AND a.id = l.k` works, `FROM a, b, l WHERE ...` is an error that says so. Or write `JOIN ... ON SEM_JOIN(...)`.
- An input column whose name contains a `.` (a quoted `"user.name"`) is rejected: the join binds its inputs by name and cannot tell it from a qualified name. Rename it in a subquery.
- If a right-hand column name collides with a left one, the join node renames it and a projection above restores the names SQL expects.

```sql
SELECT * FROM a JOIN b ON SEM_JOIN('same product?', 'm', a.title, b.name);
SELECT * FROM a, b WHERE SEM_JOIN('same product?', 'm', a.title, b.name);
```

### `GENERATE`

```
FROM GENERATE(instruction, model, 'name: type[, ...]', num_rows [, examples])
```

A source with no input: the model composes up to `num_rows` rows of the declared schema.

- `num_rows` is an integer literal of at least 1; constant expressions such as `10 + 10` are folded.
- `examples` is an optional string literal of few-shot rows: one JSON object per line, blank lines skipped. The keys must be fields of the schema, written as the (lowercased) field names; a missing key is NULL.
- Output: the declared columns, all nullable. An alias after the call works. A row the model cannot produce is dropped, so fewer than `num_rows` rows may come back.
- Named arguments, `WITH ORDINALITY`, `TABLESAMPLE` and similar table clauses on the call are errors.

```sql
SELECT * FROM GENERATE('20 fictional startups', 'm', 'name: Utf8, pitch: Utf8', 20);
SELECT * FROM GENERATE('films', 'm', 'title: Utf8, year: Int64', 5,
'{"title": "The Salt Road", "year": 2012}
{"title": "Orbit of Glass", "year": 2021}');
```

## Output types

The type and schema literals accept these spellings, in any case:

| Type | Spellings |
|---|---|
| `Utf8` | `utf8`, `text`, `string`, `varchar` |
| `Int64` | `int64`, `int`, `integer`, `bigint` |
| `Float64` | `float64`, `float`, `double`, `real` |
| `Boolean` | `boolean`, `bool` |

Field names in a schema literal are lowercased unless double-quoted: `'Title: Utf8'` declares `title`, `'"Title": Utf8'` declares `Title` (`""` inside quotes is a literal quote). Commas and colons inside a quoted name do not split fields. Duplicate names are an error after this normalisation.

## Using it on a session

`NousDBContextBuilder` installs everything by default; turn it off with `.with_sql_functions(false)` for a session whose SQL must never reach a model (the MCP server does this). Then:

```rust
let ctx = NousDBContextBuilder::new(registry).build()?;
let df = ctx.nousdb_sql("SELECT *, EXTEND('classify sentiment', 'm', review) AS s FROM reviews").await?;
```

To add the functions to a session you build yourself, call `nousdb_sql::install` on a `SessionStateBuilder` (after `with_default_features()`) or `nousdb_sql::register` on an existing `SessionContext`. Either adds three things: the marker scalar functions `extend`, `sem_filter` and `sem_join`, a relation planner for `GENERATE` and `EXTEND` in `FROM`, and an analyzer rule that rewrites the markers into NousDB's logical nodes. The session must also plan those nodes physically (a `NousDBContext` does); on a plain `SessionContext` the rewritten query fails when it is physically planned.

## Optimizer interplay

The optimizer's rewrites A to F (selection pushdown, selection reordering, join commutativity, extend-selection and extend-join fusion) fire on the plans this crate builds. A table alias or a CTE between two semantic operators adds a `SubqueryAlias` node, which stops those rewrites from firing across it; results stay correct, only the plan is less optimized. An unaliased derived table, `FROM (SELECT ...)`, adds no such node.

DataFusion does not push relational filters through the semantic nodes. A relational condition written above an `EXTEND` stays above it, between the extend and any `SEM_FILTER` above that, and rewrite A (selection pushdown past an extend) needs the selection directly above the extend. Write relational conditions in the query that holds the `EXTEND` (below it) so the model sees fewer rows, and the `SEM_FILTER` in the query above it. For the same reason, `FROM EXTEND(table, ...) WHERE ...` runs the model over the whole table (or subquery) it names; limit the input with a subquery when that matters.

## Limits in this version

- Scalar `EXTEND` in a SELECT list is computed for every input row of that SELECT, even inside a `CASE` or `COALESCE` branch SQL would not evaluate: `SELECT id, CASE WHEN id > 5 THEN EXTEND(...) END` sends every row to the model.
- A plain `LIMIT` (and `OFFSET`) above an `EXTEND` is pushed below it by DataFusion's limit pushdown, which the extend operator declares it supports, so `SELECT EXTEND(...) FROM t LIMIT 1` calls the model for one row. A `LIMIT` above `SEM_FILTER` is not pushed below it (a filter keeps an unknown share of its rows); the operator takes it as a fetch, as DataFusion's own filter does: it stops after that many accepted rows and sends a request only while the ones in flight cannot fill the limit. A `LIMIT` above `GENERATE` is taken by it as a scan takes one: `GENERATE(..., 100) LIMIT 5` asks the model for rows 1 to 5 of the 100-row table. `ORDER BY ... LIMIT` still calls the model for every row (the sort needs every row; sort and limit the input in a subquery when the sort key is an input column), and so does a `LIMIT` above `SEM_JOIN`. A relational `WHERE` above an `EXTEND` does not reduce model calls either (see "Optimizer interplay"); filter the input in a subquery.
- `SEM_FILTER` over a relational join runs on the joined rows, not the base rows (see `SEM_FILTER`).
- `SEM_JOIN` in `WHERE` over three or more comma-joined relations depends on their order in `FROM` (see `SEM_JOIN`).
- `EXTEND` inside `WHERE`, `GROUP BY` or `ORDER BY` (use a subquery).
- A struct-returning `EXTEND`; use the relation form for several columns.
- A per-call batch size or reasoning mode; both are session-level.
- Semantic SQL in the Python bindings (`nousdb-python`'s `ctx.sql` runs on the stock DataFusion session) and in the MCP server.
- Output types other than `Utf8`, `Int64`, `Float64` and `Boolean`.

## Errors

A function in an unsupported position is a plan error that names the function and where it is supported. A malformed call is a plan error with the function's usage line. In a predicate (`WHERE`, `ON`) a malformed call surfaces through DataFusion's type coercion, so the message reads `type_coercion caused by ... SEM_FILTER: expected at least 3 arguments ...`; the usage line is in it.
