"""Integration tests for faculty-managed student account status."""

import atexit
import os
import sqlite3
import tempfile
import unittest
import uuid
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if "STUDENT_DB_PATH" not in os.environ:
    _test_database_dir = tempfile.TemporaryDirectory(prefix="student-account-status-tests-")
    atexit.register(_test_database_dir.cleanup)
    os.environ["STUDENT_DB_PATH"] = os.path.join(
        _test_database_dir.name,
        "student_records.db",
    )

import app  # noqa: E402
import db  # noqa: E402


class StudentAccountStatusTests(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="student-account-status-case-")
        self.addCleanup(self.tempdir.cleanup)
        self.db_path_patch = patch.object(
            db,
            "DB_PATH",
            os.path.join(self.tempdir.name, "student_records.db"),
        )
        self.db_path_patch.start()
        self.addCleanup(self.db_path_patch.stop)
        db.init_db()
        self.client = app.app.test_client()

        faculty_login = self.client.post("/auth/login", json={
            "username": db.FACULTY_USERNAME,
            "password": db.FACULTY_PASSWORD,
        })
        self.assertEqual(faculty_login.status_code, 200)
        self.faculty_headers = {
            "Authorization": f"Bearer {faculty_login.get_json()['token']}",
        }

        self.suffix = uuid.uuid4().hex[:10]
        self.password = f"Fictional-{self.suffix}-Password"
        created = db.create_student_account(
            f"Fictional Student {self.suffix}",
            f"TEST-{self.suffix}",
            f"fictional.{self.suffix}",
            self.password,
        )
        self.student_id = created["id"]
        self.username = created["username"]
        student_login = self.client.post("/auth/login", json={
            "username": self.username,
            "password": self.password,
        })
        self.assertEqual(student_login.status_code, 200)
        self.student_token = student_login.get_json()["token"]
        self.student_headers = {
            "Authorization": f"Bearer {self.student_token}",
        }

    def set_active(self, is_active):
        return self.client.put(
            f"/students/{self.student_id}/account-status",
            headers=self.faculty_headers,
            json={"is_active": is_active},
        )

    def test_faculty_can_deactivate_and_reactivate_account_idempotently(self):
        deactivated = self.set_active(False)
        self.assertEqual(deactivated.status_code, 200)
        self.assertFalse(deactivated.get_json()["account_active"])

        inactive_listing = self.client.get("/students", headers=self.faculty_headers)
        inactive_student = next(
            item for item in inactive_listing.get_json()["students"]
            if item["id"] == self.student_id
        )
        self.assertFalse(inactive_student["account_active"])

        with db.get_connection() as connection:
            faculty_active = connection.execute(
                "SELECT is_active FROM users WHERE role = 'faculty' LIMIT 1"
            ).fetchone()["is_active"]
        self.assertEqual(faculty_active, 1)

        repeated_deactivation = self.set_active(False)
        self.assertEqual(repeated_deactivation.status_code, 200)
        self.assertFalse(repeated_deactivation.get_json()["account_active"])

        reactivated = self.set_active(True)
        self.assertEqual(reactivated.status_code, 200)
        self.assertTrue(reactivated.get_json()["account_active"])

        repeated_reactivation = self.set_active(True)
        self.assertEqual(repeated_reactivation.status_code, 200)
        self.assertTrue(repeated_reactivation.get_json()["account_active"])

        listing = self.client.get("/students", headers=self.faculty_headers)
        self.assertEqual(listing.status_code, 200)
        student = next(
            item for item in listing.get_json()["students"]
            if item["id"] == self.student_id
        )
        self.assertTrue(student["account_active"])

        login = self.client.post("/auth/login", json={
            "username": self.username,
            "password": self.password,
        })
        self.assertEqual(login.status_code, 200)

    def test_status_change_requires_faculty_authentication(self):
        missing_auth = self.client.put(
            f"/students/{self.student_id}/account-status",
            json={"is_active": False},
        )
        self.assertEqual(missing_auth.status_code, 401)

        student_request = self.client.put(
            f"/students/{self.student_id}/account-status",
            headers=self.student_headers,
            json={"is_active": False},
        )
        self.assertEqual(student_request.status_code, 403)

        login_after_denied_request = self.client.post("/auth/login", json={
            "username": self.username,
            "password": self.password,
        })
        self.assertEqual(login_after_denied_request.status_code, 200)

    def test_missing_students_and_accounts_return_not_found(self):
        missing_student = self.client.put(
            "/students/999999/account-status",
            headers=self.faculty_headers,
            json={"is_active": False},
        )
        self.assertEqual(missing_student.status_code, 404)

        with db.get_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO students (roll_no, name, created_at) VALUES (?, ?, ?)",
                (f"NO-ACCOUNT-{self.suffix}", "Fictional Record Without Account", db.utc_now()),
            )
            student_without_account = cursor.lastrowid

        missing_account = self.client.put(
            f"/students/{student_without_account}/account-status",
            headers=self.faculty_headers,
            json={"is_active": False},
        )
        self.assertEqual(missing_account.status_code, 404)
        self.assertIn("Student account not found", missing_account.get_json()["message"])

    def test_invalid_status_payloads_are_rejected(self):
        invalid_payloads = [
            None,
            {},
            {"is_active": "false"},
            {"is_active": 0},
            {"is_active": None},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.put(
                    f"/students/{self.student_id}/account-status",
                    headers=self.faculty_headers,
                    json=payload,
                )
                self.assertEqual(response.status_code, 400)

    def test_deactivation_rejects_login_and_invalidates_existing_tokens(self):
        with db.get_connection() as connection:
            token_count_before = connection.execute(
                "SELECT COUNT(*) FROM auth_tokens WHERE user_id = "
                "(SELECT id FROM users WHERE student_id = ?)",
                (self.student_id,),
            ).fetchone()[0]
        self.assertEqual(token_count_before, 1)

        self.assertEqual(self.set_active(False).status_code, 200)

        with db.get_connection() as connection:
            token_count_after = connection.execute(
                "SELECT COUNT(*) FROM auth_tokens WHERE user_id = "
                "(SELECT id FROM users WHERE student_id = ?)",
                (self.student_id,),
            ).fetchone()[0]
        self.assertEqual(token_count_after, 0)

        for endpoint in (
            "/auth/me",
            f"/students/{self.student_id}",
            f"/students/{self.student_id}/predictions",
            f"/students/{self.student_id}/interventions",
        ):
            with self.subTest(endpoint=endpoint):
                response = self.client.get(endpoint, headers=self.student_headers)
                self.assertEqual(response.status_code, 401)

        login = self.client.post("/auth/login", json={
            "username": self.username,
            "password": self.password,
        })
        self.assertEqual(login.status_code, 401)
        self.assertEqual(
            login.get_json()["message"],
            "Invalid username or password.",
        )

    def test_deactivation_preserves_academic_prediction_intervention_and_audit_data(self):
        with db.get_connection() as connection:
            student_user_id = connection.execute(
                "SELECT id FROM users WHERE student_id = ?",
                (self.student_id,),
            ).fetchone()["id"]
            faculty_user_id = connection.execute(
                "SELECT id FROM users WHERE role = 'faculty' LIMIT 1"
            ).fetchone()["id"]
            connection.execute(
                "INSERT INTO academic_records (student_id, attendance, test1, test2, "
                "assignment, study_hours, faculty_updated_by, faculty_updated_at) "
                "VALUES (?, 88, 32, 34, 9, 4, ?, ?)",
                (self.student_id, faculty_user_id, db.utc_now()),
            )
            prediction_cursor = connection.execute(
                "INSERT INTO predictions (student_id, attendance, test1, test2, "
                "assignment, study_hours, predicted_marks, risk_code, risk_level, "
                "risk_basis, base_value, shap_json, recommendations_json, "
                "created_by_user_id, created_by_role, created_at) "
                "VALUES (?, 88, 32, 34, 9, 4, 84, 'lower_concern', 'Lower Concern', "
                "'predicted_marks', 64, '{}', '[]', ?, 'student', ?)",
                (self.student_id, student_user_id, db.utc_now()),
            )
            prediction_id = prediction_cursor.lastrowid
            connection.execute(
                "INSERT INTO interventions (student_id, prediction_id, title, "
                "description, category, status, created_by_user_id, created_at) "
                "VALUES (?, ?, 'Fictional task', 'Preservation test', 'Study Habits', "
                "'assigned', ?, ?)",
                (self.student_id, prediction_id, faculty_user_id, db.utc_now()),
            )
            before = {
                "academic": tuple(connection.execute(
                    "SELECT attendance, test1, test2, assignment, study_hours, "
                    "faculty_updated_by, faculty_updated_at "
                    "FROM academic_records WHERE student_id = ?",
                    (self.student_id,),
                ).fetchone()),
                "prediction": tuple(connection.execute(
                    "SELECT student_id, predicted_marks, shap_json, recommendations_json, "
                    "created_by_user_id, created_by_role "
                    "FROM predictions WHERE id = ?",
                    (prediction_id,),
                ).fetchone()),
                "intervention": tuple(connection.execute(
                    "SELECT student_id, prediction_id, title, description, category, "
                    "created_by_user_id FROM interventions WHERE student_id = ?",
                    (self.student_id,),
                ).fetchone()),
            }

        self.assertEqual(self.set_active(False).status_code, 200)
        self.assertEqual(self.set_active(True).status_code, 200)

        with db.get_connection() as connection:
            after = {
                "academic": tuple(connection.execute(
                    "SELECT attendance, test1, test2, assignment, study_hours, "
                    "faculty_updated_by, faculty_updated_at "
                    "FROM academic_records WHERE student_id = ?",
                    (self.student_id,),
                ).fetchone()),
                "prediction": tuple(connection.execute(
                    "SELECT student_id, predicted_marks, shap_json, recommendations_json, "
                    "created_by_user_id, created_by_role "
                    "FROM predictions WHERE id = ?",
                    (prediction_id,),
                ).fetchone()),
                "intervention": tuple(connection.execute(
                    "SELECT student_id, prediction_id, title, description, category, "
                    "created_by_user_id FROM interventions WHERE student_id = ?",
                    (self.student_id,),
                ).fetchone()),
            }
        self.assertEqual(after, before)

    def test_startup_migration_preserves_existing_accounts_and_defaults_them_active(self):
        legacy_db_path = os.path.join(self.tempdir.name, "legacy.db")
        connection = sqlite3.connect(legacy_db_path)
        try:
            connection.executescript("""
                CREATE TABLE students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    roll_no TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    program TEXT,
                    section TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('faculty', 'student')),
                    student_id INTEGER UNIQUE REFERENCES students (id),
                    created_at TEXT NOT NULL,
                    CHECK (
                        (role = 'student' AND student_id IS NOT NULL)
                        OR (role = 'faculty' AND student_id IS NULL)
                    )
                );
                INSERT INTO students (id, roll_no, name, created_at)
                    VALUES (41, 'LEGACY-41', 'Legacy Fictional Student', '2025-01-01');
                INSERT INTO users (id, username, password_hash, role, student_id, created_at)
                    VALUES (73, 'legacy.user', 'existing-password-hash', 'student', 41, '2025-01-01');
            """)
        finally:
            connection.close()

        with patch.object(db, "DB_PATH", legacy_db_path):
            db.init_db()
            db.init_db()
            with db.get_connection() as connection:
                account = connection.execute(
                    "SELECT id, username, password_hash, role, student_id, is_active "
                    "FROM users WHERE id = 73"
                ).fetchone()
                student_count = connection.execute(
                    "SELECT COUNT(*) FROM students"
                ).fetchone()[0]
                account_count = connection.execute(
                    "SELECT COUNT(*) FROM users"
                ).fetchone()[0]

        self.assertEqual(
            tuple(account),
            (73, "legacy.user", "existing-password-hash", "student", 41, 1),
        )
        self.assertEqual(student_count, 1)
        self.assertEqual(account_count, 1)


if __name__ == "__main__":
    unittest.main()
