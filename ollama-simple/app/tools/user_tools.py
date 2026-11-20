
from database.connection import get_connection


def create_user(name: str, email: str) -> dict:
    """
    Creates a new user in the database.
    """

    if not name.strip() or "@" not in email:
        return {
            "success": False,
            "message": "Invalid name or email."
        }

    try:
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE
                )
            """)

            cursor = conn.execute(
                """
                INSERT INTO users (name, email)
                VALUES (?, ?)
                """,
                (name, email)
            )

            return {
                "success": True,
                "user_id": cursor.lastrowid,
                "name": name,
                "email": email
            }

    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            return {
                "success": False,
                "message": "Email already registered."
            }

        raise


def list_users() -> dict:
    """
    Returns all registered users.
    """

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE
            )
        """)

        users = conn.execute(
            "SELECT id, name, email FROM users ORDER BY id"
        ).fetchall()

        return {
            "success": True,
            "total": len(users),
            "users": [dict(user) for user in users]
        }
