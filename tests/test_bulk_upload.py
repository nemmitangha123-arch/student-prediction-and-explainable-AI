"""Integration tests for the faculty bulk CSV import workflow."""

import io
import atexit
import os
import tempfile
import unittest
import uuid
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if "STUDENT_DB_PATH" not in os.environ:
    _test_database_dir = tempfile.TemporaryDirectory(prefix="student-bulk-tests-")
    atexit.register(_test_database_dir.cleanup)
    os.environ["STUDENT_DB_PATH"] = os.path.join(
        _test_database_dir.name,
        "student_records.db"
    )

import app  # noqa: E402
import db  # noqa: E402
from werkzeug.security import generate_password_hash  # noqa: E402


HEADERS = [
    "Student ID",
    "Student Name",
    "Attendance",
    "Internal Test 1",
    "Internal Test 2",
    "Assignment Score",
    "Daily Study Hours",
]


class BulkUploadTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()
        response = cls.client.post("/auth/login", json={
            "username": db.FACULTY_USERNAME,
            "password": db.FACULTY_PASSWORD,
        })
        cls.faculty_token = response.get_json()["token"]
        cls.faculty_headers = {"Authorization": f"Bearer {cls.faculty_token}"}

    def setUp(self):
        suffix = uuid.uuid4().hex[:12]
        self.student_id = f"BULK-{suffix}"
        self.student_name = f"Bulk Student {suffix}"

        with db.get_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO students (roll_no, name, program, section, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (self.student_id, self.student_name, "B.Tech CSE", "A", db.utc_now())
            )
            self.database_student_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO academic_records (student_id) VALUES (?)",
                (self.database_student_id,)
            )
            connection.execute(
                "INSERT INTO users (username, password_hash, role, student_id, created_at) "
                "VALUES (?, ?, 'student', ?, ?)",
                (
                    self.student_id,
                    generate_password_hash("Test-Student-Password"),
                    self.database_student_id,
                    db.utc_now()
                )
            )

    @staticmethod
    def csv_bytes(rows, headers=None):
        columns = headers or HEADERS
        lines = [",".join(columns)]
        lines.extend(",".join(str(value) for value in row) for row in rows)
        return ("\n".join(lines) + "\n").encode("utf-8")

    def row(self, student_id=None, student_name=None, **overrides):
        values = {
            "Student ID": student_id or self.student_id,
            "Student Name": student_name or self.student_name,
            "Attendance": 91,
            "Internal Test 1": 32,
            "Internal Test 2": 36,
            "Assignment Score": 7,
            "Daily Study Hours": 3,
        }
        values.update(overrides)
        return [values[header] for header in HEADERS]

    def upload(self, endpoint, content, filename="students.csv", headers=None):
        request_headers = self.faculty_headers if headers is None else headers
        return self.client.post(
            endpoint,
            data={"file": (io.BytesIO(content), filename)},
            headers=request_headers,
            content_type="multipart/form-data"
        )

    def test_valid_csv_imports_multiple_records_with_predictions_and_history(self):
        second_id = f"BULK-{uuid.uuid4().hex[:12]}"
        second_name = f"Bulk Student {second_id[-6:]}"
        with db.get_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO students (roll_no, name, program, section, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (second_id, second_name, "B.Tech CSE", "A", db.utc_now())
            )
            second_database_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO academic_records (student_id) VALUES (?)",
                (second_database_id,)
            )
            connection.execute(
                "INSERT INTO users (username, password_hash, role, student_id, created_at) "
                "VALUES (?, ?, 'student', ?, ?)",
                (
                    second_id,
                    generate_password_hash("Test-Student-Password"),
                    second_database_id,
                    db.utc_now()
                )
            )

        rows = [
            self.row(),
            self.row(second_id, second_name, Attendance=68, **{
                "Internal Test 1": 24,
                "Internal Test 2": 25,
                "Assignment Score": 6,
                "Daily Study Hours": 2,
            }),
        ]
        content = self.csv_bytes(rows)

        preview = self.upload("/students/bulk-upload/validate", content)
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.get_json()["valid_rows"], 2)
        self.assertEqual(preview.get_json()["rejected_rows"], 0)

        response = self.upload("/students/bulk-upload/import", content)
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["total_rows"], 2)
        self.assertEqual(payload["successful_imports"], 2)
        self.assertEqual(payload["rejected_rows"], 0)

        for item in payload["results"]:
            with self.subTest(student_id=item["student_id"]):
                inputs = next(
                    record["inputs"]
                    for record in preview.get_json()["rows"]
                    if record["student_id"] == item["student_id"]
                )
                expected = app.analyze_inputs(inputs)
                self.assertEqual(item["predicted_marks"], round(expected["predicted_marks"], 2))
                self.assertEqual(item["risk_code"], expected["risk_code"])
                self.assertEqual(item["risk_level"], expected["risk_level"])

                student = db.get_student(
                    self.database_student_id
                    if item["student_id"] == self.student_id
                    else second_database_id
                )
                history = db.list_predictions(student["id"])
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]["risk_code"], expected["risk_code"])
                self.assertEqual(history[0]["created_by_role"], "faculty")
                self.assertEqual(history[0]["shap_contributions"], expected["shap_contributions"])

        imported = payload["results"][0]
        prediction_id = imported["prediction_id"]
        intervention = self.client.post(
            f"/students/{self.database_student_id}/interventions",
            headers=self.faculty_headers,
            json={
                "title": "Follow up on prediction",
                "description": "Review the saved recommendation.",
                "category": "General",
                "prediction_id": prediction_id,
            }
        )
        self.assertEqual(intervention.status_code, 201)
        self.assertEqual(intervention.get_json()["intervention"]["prediction_id"], prediction_id)

        student_login = self.client.post("/auth/login", json={
            "username": self.student_id,
            "password": "Test-Student-Password",
        })
        student_token = student_login.get_json()["token"]
        student_detail = self.client.get(
            f"/students/{self.database_student_id}",
            headers={"Authorization": f"Bearer {student_token}"}
        )
        self.assertEqual(student_detail.status_code, 200)
        self.assertEqual(student_detail.get_json()["prediction_status"]["status"], "current")

    def test_header_variations_and_missing_required_headers(self):
        normalized_headers = [
            "student_id",
            "Student Name",
            "Attendance (%)",
            "Internal Test 1 (out of 40)",
            "test2",
            "Assignment Score",
            "Daily Study Hours",
        ]
        response = self.upload(
            "/students/bulk-upload/validate",
            self.csv_bytes([self.row()], normalized_headers)
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["valid_rows"], 1)

        missing = self.upload(
            "/students/bulk-upload/validate",
            self.csv_bytes([self.row()], HEADERS[:-1])
        )
        self.assertEqual(missing.status_code, 400)
        self.assertTrue(any(error["field"] == "study_hours" for error in missing.get_json()["errors"]))

    def test_missing_values_and_invalid_numeric_inputs_are_row_errors(self):
        invalid = self.row(Attendance="", **{"Internal Test 1": "not-a-number"})
        boolean = self.row(
            student_id=f"BOOL-{uuid.uuid4().hex[:8]}",
            student_name="Boolean Input",
            **{"Daily Study Hours": "True"}
        )
        response = self.upload(
            "/students/bulk-upload/validate",
            self.csv_bytes([invalid, boolean])
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["valid_rows"], 0)
        fields = {error["field"] for error in body["errors"]}
        self.assertIn("attendance", fields)
        self.assertIn("test1", fields)
        self.assertTrue(any(
            error["field"] == "study_hours" and "numeric" in error["reason"]
            for error in body["errors"]
        ))

    def test_out_of_range_values_are_rejected(self):
        response = self.upload(
            "/students/bulk-upload/validate",
            self.csv_bytes([self.row(Attendance=51, **{"Daily Study Hours": 6})])
        )
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["rejected_rows"], 1)
        self.assertEqual(
            {error["field"] for error in body["errors"]},
            {"attendance", "study_hours"}
        )

    def test_duplicate_ids_reject_both_rows(self):
        response = self.upload(
            "/students/bulk-upload/validate",
            self.csv_bytes([self.row(), self.row()])
        )
        body = response.get_json()
        self.assertEqual(body["valid_rows"], 0)
        self.assertEqual(body["rejected_rows"], 2)
        self.assertEqual(
            [row["status"] for row in body["rows"]],
            ["invalid", "invalid"]
        )

    def test_existing_records_name_conflicts_and_unknown_accounts_are_rejected(self):
        existing = ["PBL-2026-001", "Aarav Mehta", 91, 32, 36, 7, 3]
        mismatch = self.row(student_name="A different person")
        unknown = self.row(student_id="NOT-IN-DATABASE", student_name="Nobody")

        no_account_id = f"NO-ACCOUNT-{uuid.uuid4().hex[:8]}"
        with db.get_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO students (roll_no, name, program, section, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (no_account_id, "No Login", None, None, db.utc_now())
            )
            connection.execute(
                "INSERT INTO academic_records (student_id) VALUES (?)",
                (cursor.lastrowid,)
            )
        no_account = self.row(student_id=no_account_id, student_name="No Login")

        response = self.upload(
            "/students/bulk-upload/validate",
            self.csv_bytes([existing, mismatch, unknown, no_account])
        )
        body = response.get_json()
        self.assertEqual(body["valid_rows"], 0)
        reasons = " ".join(error["reason"] for error in body["errors"])
        self.assertIn("already exists", reasons)
        self.assertIn("does not match", reasons)
        self.assertIn("existing student record", reasons)
        self.assertIn("student account", reasons)

    def test_upload_requires_authentication_and_faculty_role(self):
        content = self.csv_bytes([self.row()])
        student_login = self.client.post("/auth/login", json={
            "username": "PBL-2026-001",
            "password": "Student@001",
        })
        token = student_login.get_json()["token"]
        for endpoint in (
            "/students/bulk-upload/validate",
            "/students/bulk-upload/import",
        ):
            with self.subTest(endpoint=endpoint):
                unauthenticated = self.upload(endpoint, content, headers={})
                self.assertEqual(unauthenticated.status_code, 401)
                unauthorized = self.upload(
                    endpoint,
                    content,
                    headers={"Authorization": f"Bearer {token}"}
                )
                self.assertEqual(unauthorized.status_code, 403)

    def test_database_failure_rolls_back_all_student_records_and_predictions(self):
        second_id = f"BULK-{uuid.uuid4().hex[:12]}"
        second_name = f"Bulk Student {second_id[-6:]}"
        with db.get_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO students (roll_no, name, program, section, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (second_id, second_name, None, None, db.utc_now())
            )
            second_database_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO academic_records (student_id) VALUES (?)",
                (second_database_id,)
            )
            connection.execute(
                "INSERT INTO users (username, password_hash, role, student_id, created_at) "
                "VALUES (?, ?, 'student', ?, ?)",
                (second_id, "unused-hash", second_database_id, db.utc_now())
            )

        original_insert = db._insert_prediction
        calls = 0

        def fail_on_second_insert(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated database failure")
            return original_insert(*args, **kwargs)

        with patch.object(db, "_insert_prediction", side_effect=fail_on_second_insert):
            response = self.upload(
                "/students/bulk-upload/import",
                self.csv_bytes([self.row(), self.row(second_id, second_name)])
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["successful_imports"], 0)
        self.assertIn("No records were imported", response.get_json()["message"])

        for student_id in (self.database_student_id, second_database_id):
            student = db.get_student(student_id)
            self.assertIsNone(student["record"]["attendance"])
            self.assertEqual(db.list_predictions(student_id), [])

    def test_stale_preview_conflict_rolls_back_batch(self):
        content = self.csv_bytes([self.row()])
        preview = self.upload("/students/bulk-upload/validate", content)
        self.assertEqual(preview.get_json()["valid_rows"], 1)

        inspect_students = db.inspect_bulk_students

        def change_after_preview(student_ids):
            result = inspect_students(student_ids)
            db.update_faculty_marks(
                self.database_student_id,
                attendance=90,
                test1=30,
                test2=30,
                assignment=7
            )
            return result

        with patch.object(db, "inspect_bulk_students", side_effect=change_after_preview):
            response = self.upload("/students/bulk-upload/import", content)

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["successful_imports"], 0)
        student = db.get_student(self.database_student_id)
        self.assertEqual(student["record"]["attendance"], 90)
        self.assertEqual(db.list_predictions(self.database_student_id), [])

    def test_prediction_failure_reports_row_and_saves_nothing(self):
        with patch.object(app, "analyze_inputs", side_effect=RuntimeError("model failure")):
            response = self.upload(
                "/students/bulk-upload/import",
                self.csv_bytes([self.row()])
            )

        self.assertEqual(response.status_code, 500)
        payload = response.get_json()
        self.assertEqual(payload["successful_imports"], 0)
        self.assertEqual(payload["rejected_rows"], 1)
        self.assertEqual(payload["errors"][0]["row_number"], 2)
        self.assertEqual(payload["errors"][0]["field"], "prediction")
        self.assertIsNone(db.get_student(self.database_student_id)["record"]["attendance"])
        self.assertEqual(db.list_predictions(self.database_student_id), [])

    def test_empty_malformed_oversized_and_unsupported_files(self):
        empty = self.upload("/students/bulk-upload/validate", b"")
        self.assertEqual(empty.status_code, 400)

        malformed = self.upload(
            "/students/bulk-upload/validate",
            (",".join(HEADERS) + '\n"unterminated\n').encode("utf-8")
        )
        self.assertEqual(malformed.status_code, 400)

        oversized = self.upload(
            "/students/bulk-upload/validate",
            b"x" * (2 * 1024 * 1024 + 1)
        )
        self.assertEqual(oversized.status_code, 400)

        request_too_large = self.upload(
            "/students/bulk-upload/validate",
            b"x" * 2_300_000
        )
        self.assertEqual(request_too_large.status_code, 413)

        unsupported = self.upload(
            "/students/bulk-upload/validate",
            self.csv_bytes([self.row()]),
            filename="students.xlsx"
        )
        self.assertEqual(unsupported.status_code, 400)

        too_many_rows = self.upload(
            "/students/bulk-upload/validate",
            self.csv_bytes([self.row()] * 501)
        )
        self.assertEqual(too_many_rows.status_code, 400)

    def test_existing_prediction_and_dashboard_workflows_remain_available(self):
        db.update_faculty_marks(
            self.database_student_id,
            attendance=91,
            test1=32,
            test2=36,
            assignment=7
        )
        db.update_study_hours(self.database_student_id, 3)
        result = self.client.post(
            f"/students/{self.database_student_id}/predictions",
            headers=self.faculty_headers
        )
        self.assertEqual(result.status_code, 201)
        detail = self.client.get(
            f"/students/{self.database_student_id}",
            headers=self.faculty_headers
        )
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.get_json()["prediction_status"]["status"], "current")
        listing = self.client.get("/students", headers=self.faculty_headers)
        self.assertEqual(listing.status_code, 200)
        listed = next(
            student
            for student in listing.get_json()["students"]
            if student["id"] == self.database_student_id
        )
        self.assertEqual(listed["prediction_status"]["status"], "current")


if __name__ == "__main__":
    unittest.main()
