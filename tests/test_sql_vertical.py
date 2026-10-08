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


def test_sql_model_uses_common_benchmark_view_with_private_assessment_deferred():
    import json
    from dataclasses import asdict

    from mindscape.data.schemas import BenchmarkExample
    from mindscape.sql.policy import SqlBenchmarkModel

    example = task(0)
    env = SqlEnvironment()
    initial = asdict(env.reset(example))
    benchmark = BenchmarkExample(
        example.task_id,
        "sql_correction",
        example.problem,
        example.visible(),
        initial,
        {"description": example.problem},
        example.target_query,
        [],
        {"private_rows": example.private_rows},
    )

    class Coder:
        calls = 0
        parameter_count = 1

        def generate(self, system, prompt, *args):
            payload = json.loads(prompt)
            assert "private_rows" not in payload and "target_query" not in payload
            self.calls += 1
            return json.dumps({"query": example.target_query})

    prediction = SqlBenchmarkModel(Coder(), structured=False).predict(
        benchmark.view("experiential")
    )
    assert prediction.diagnostics["assessment"] == "deferred"
    env.query = prediction.answer
    assert env.final_evaluate()["success"]
