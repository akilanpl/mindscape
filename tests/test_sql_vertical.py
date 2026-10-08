from mindscape.sql.environment import SqlAction, SqlEnvironment, execute, task


def test_sql_actual_goal_and_private_boundary():
    for index in range(5):
        example = task(index)
        env = SqlEnvironment()
        env.reset(example)
        assert "private_rows" not in example.visible() and "target_query" not in example.visible()
        assert not env.final_evaluate()["success"]
        env.step(SqlAction("edit_query", example.target_query))
        transition = env.step(SqlAction("execute_query"))
        assert transition.valid and transition.result["actual_result"]
        assert env.final_evaluate()["success"]


def test_sql_capabilities_and_resource_bound():
    example = task(0)
    for query in (
        "ATTACH DATABASE '/tmp/exfil' AS other",
        "DROP TABLE items",
        "DELETE FROM items",
        "SELECT load_extension('/tmp/evil')",
        "PRAGMA database_list",
    ):
        assert execute(example.schema, example.visible_rows, query)["error"]
    query = "WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x"
    assert execute(example.schema, example.visible_rows, query, timeout=0.01)["error"]
