import sqlite3
import time
from dataclasses import dataclass

from mindscape.core.schema import Entity, Goal, Relation


@dataclass(frozen=True)
class SqlTask:
    task_id: str
    problem: str
    query: str
    schema: str
    visible_rows: tuple
    private_rows: tuple
    target_query: str

    def visible(self):
        return {
            "task_id": self.task_id,
            "problem": self.problem,
            "query": self.query,
            "schema": self.schema,
            "visible_rows": self.visible_rows,
        }


@dataclass(frozen=True)
class SqlState:
    query: str
    goal: Goal
    entities: tuple
    relations: tuple
    observations: tuple
    previous_actions: tuple


@dataclass(frozen=True)
class SqlAction:
    name: str
    query: str | None = None


@dataclass(frozen=True)
class SqlTransition:
    state_before: SqlState
    action: SqlAction
    event: str
    result: dict
    state_after: SqlState
    valid: bool


def execute(schema, rows, query, timeout=0.2, max_rows=100):
    """No files/extensions/subprocesses; SELECT capability and bounded VM work."""
    connection = sqlite3.connect(":memory:")
    began = time.perf_counter()
    try:
        connection.executescript(schema)
        connection.executemany("INSERT INTO items VALUES (?, ?, ?)", rows)
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }
        connection.set_authorizer(
            lambda action, *unused: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        )
        connection.set_progress_handler(lambda: int(time.perf_counter() - began > timeout), 1000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        cursor = connection.execute(query)
        values = cursor.fetchmany(max_rows + 1)
        if len(values) > max_rows:
            raise ValueError("Row budget exceeded")
        return {
            "rows": values,
            "error": None,
            "actual_result": True,
            "seconds": time.perf_counter() - began,
        }
    except (sqlite3.Error, ValueError) as error:
        return {
            "rows": None,
            "error": str(error),
            "actual_result": True,
            "seconds": time.perf_counter() - began,
        }
    finally:
        connection.close()


class SqlEnvironment:
    tools = ("inspect_schema", "inspect_table", "edit_query", "execute_query", "finish")

    def reset(self, task):
        self.task = task
        self.query = task.query
        self.observations = []
        self.actions = []
        self.transitions = []
        return self.get_state()

    def get_state(self):
        return SqlState(
            self.query,
            Goal(self.task.problem),
            (Entity("items", len(self.task.visible_rows)),),
            (Relation("query", "reads", "items"),),
            tuple(self.observations),
            tuple(self.actions),
        )

    def step(self, action):
        before = self.get_state()
        valid = True
        if action.name == "inspect_schema":
            result = {"schema": self.task.schema}
        elif action.name == "inspect_table":
            result = {"visible_rows": self.task.visible_rows}
        elif (
            action.name == "edit_query"
            and isinstance(action.query, str)
            and len(action.query) < 10000
        ):
            self.query = action.query
            result = {"query": self.query}
        elif action.name == "execute_query":
            result = execute(self.task.schema, self.task.visible_rows, self.query)
        elif action.name == "finish":
            result = {"stopped": True}
        else:
            valid = False
            result = {"error": "Invalid typed SQL action"}
        self.actions.append(action.name)
        self.observations.append(result)
        event = "execution_error" if result.get("error") else "action_completed"
        transition = SqlTransition(before, action, event, result, self.get_state(), valid)
        self.transitions.append(transition)
        return transition

    def final_evaluate(self):
        # Private rows and target query never enter learner observations.
        expected = execute(self.task.schema, self.task.private_rows, self.task.target_query)
        actual = execute(self.task.schema, self.task.private_rows, self.query)
        return {
            "success": expected["error"] is None
            and actual["error"] is None
            and actual["rows"] == expected["rows"],
            "actual": actual,
            "expected": expected,
        }


def task(index):
    k = 2 + index % 7
    rows = tuple((i, i * k, "a" if i % 2 else "b") for i in range(1, 7))
    private = tuple((i, i * k, "a" if i % 2 else "b") for i in range(11))
    specs = (
        (
            f"Return id and value for items whose value is strictly greater than {3 * k}, ordered by id.",
            f"SELECT id,value FROM items WHERE value > {3 * k} ORDER BY id",
            f"SELECT id,value FROM items WHERE value >= {3 * k} ORDER BY id",
        ),
        (
            "Return each category and its total value, ordered by category.",
            "SELECT category,SUM(value) FROM items GROUP BY category ORDER BY category",
            "SELECT category,COUNT(value) FROM items GROUP BY category ORDER BY category",
        ),
        (
            "Return the two items with highest values as id,value, highest first.",
            "SELECT id,value FROM items ORDER BY value DESC,id LIMIT 2",
            "SELECT id,value FROM items ORDER BY value ASC,id LIMIT 2",
        ),
        (
            "Return the count of items in category a.",
            "SELECT COUNT(*) FROM items WHERE category = 'a'",
            "SELECT COUNT(*) FROM items WHERE category = 'b'",
        ),
        (
            "Return category and average value for categories whose average exceeds 5, ordered by category.",
            "SELECT category,AVG(value) FROM items GROUP BY category HAVING AVG(value)>5 ORDER BY category",
            "SELECT category,AVG(value) FROM items WHERE AVG(value)>5 GROUP BY category ORDER BY category",
        ),
    )
    problem, target, buggy = specs[index % len(specs)]
    return SqlTask(
        "sql_" + str(index),
        problem,
        buggy,
        "CREATE TABLE items(id INTEGER,value INTEGER,category TEXT);",
        rows,
        private,
        target,
    )
